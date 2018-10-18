---
title: Tensorflow LSTM for Language Modelling
---

In this post, I will show you how to build an LSTM network for the task of character-based language modelling (predict the next character based on the previous ones), and apply it to generate lyrics. In general, the model allows you to generate new text as well as auto-complete your own sentences based on any given text database.

The code
--------

The code is available [here](https://github.com/f90/Tensorflow-Char-LSTM/tree/master), and the lyrics database can be created by running the crawler I developed [here](https://github.com/f90/lyrics-crawler).

Want to see what lyrics my model composes? [Click here!](#example-output)
------------------------------------------------------------------------

It is implemented in Tensorflow, which has been rapidly evolving in the last few months. As a result, best practices for common tasks are changing as well. That is why I built my own code to make the best use out of [Queues](https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues/index.html), and new functionality from the [training-contrib](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/contrib.training.md) package. In particular, implementing batch-training with variable-length sequences on an unrolled LSTM with truncated BPTT with maintaining the hidden state for each sequence is greatly simplified and optimised. Without further ado, let's get started!

Input pipeline
==============

First, we are going to load the dataset. I adapted this a bit to my case, but it should be easy for you to change it to your liking:

    data, vocab = Dataset.readLyrics(data_settings["input_csv"], data_settings["input_vocab"])
    trainIndices, testIndices = Dataset.createPartition(data, train_settings["trainPerc"])

`Dataset` is a helper file that manages dataset-related operations. Here I assume `data` has the form of a list of entries, each again a list with two entries: The first entry denotes the artist and the second the lyrics content. The first entry is used to prevent having the same artist in both training and test set. `trainIndices` and `testIndices` are lists of indices refer to the rows in `data` that correspond to the training and test set, respectively. The settings variables are just dictionaries that yield user-defined settings. `vocab` is a special Vocabulary object that translates characters into integer indices and vice versa, and we will get to know its functionality better along the way. For creating the batches to train on, we will use `batch_sequences_with_states`, as it is very convenient to use. It requires a key, the sequence length, and input and output for a SINGLE sequence in symbolic form. We create these properties here as placeholders, to later feed them with our own input thread. This design makes input processing very fast and ensures it does not reduce overall training speed: Model training and input processing are running simultaneously in different threads.

    keyInput = tf.placeholder(tf.string) # To identify each sequence
    lengthInput = tf.placeholder(tf.int32) # Length of sequence
    seqInput = tf.placeholder(tf.int32, shape=[None]) # Input sequence
    seqOutput = tf.placeholder(tf.int32, shape=[None]) # Output sequence

Then we create a `RandomShuffleQueue` and the enqueue and dequeue operations, which means sequences will be randomly selected from the queue to form batches during training. This presents an effective compromise between completely random sample selection, which is often slow for large datasets as data has to be pulled from very different memory locations, and completely sequential reading. Using the dictionaries ensures compatibility with the Tensorflow sequence format:

    q = tf.RandomShuffleQueue(input_settings["queue_capacity"], input_settings["min_queue_capacity"],
                          [tf.string, tf.int32, tf.int32, tf.int32])
    enqueue_op = q.enqueue([keyInput, lengthInput, seqInput, seqOutput])

    with tf.device("/cpu:0"):
        key, contextT, sequenceIn, sequenceOut = q.dequeue()
        context = {"length" : tf.reshape(contextT, [])}
        sequences = {"inputs" : tf.reshape(sequenceIn, [contextT]),
                    "outputs" : tf.reshape(sequenceOut, [contextT])}

Instead of using the built-in CSV or TFRecord Readers to enqueue samples, I created my own method that can read directly from the `data` in the RAM. It endlessly loops over the samples given by `indices` and adds them to the queue. It can easily be adapted to read arbitrary files/parts of datasets and perform further preprocessing:

    # Enqueueing method in different thread, loading sequence examples and feeding into FIFO Queue
    def load_and_enqueue(indices):
        run = True
        key = 0 # Unique key for every sample, even over multiple epochs (otherwise the queue could be filled up with two same-key examples)
        while run:
            for index in indices:
                current_seq = data[index][1]
                try:
                    sess.run(enqueue_op, feed_dict={keyInput: str(key),
                                              lengthInput: len(current_seq)-1,
                                            seqInput: current_seq[:-1],
                                            seqOutput: current_seq[1:]},
                                    options=tf.RunOptions(timeout_in_ms=60000))
                except tf.errors.DeadlineExceededError as e:
                    print("Timeout while waiting to enqueue into input queue! Stopping input queue thread!")
                    run = False
                    break
                key += 1
            print "Finished enqueueing all " + str(len(indices)) + " samples!"

`load_and_enqueue` will be started as a separate Thread later. Two important things to note here. One is the `key` which is different even for the same samples to avoid errors when incidentally queueing the same samples into the queue, which causes the keys to clash. The other is the timeout, which is the only way I found to be able to stop the training by closing the queue and then catching the resulting `DeadlineExceededError`. Lastly, the input and output is delayed by one step to force the model to predict the upcoming character.

The LSTM model
==============

Our model is an LSTM with a variable number of layers and configurable dropout, whose hidden states will be maintained separately for each sequence. I created a new class `LyricsPredictor` with an inference method that builds the computational graph. The beginning looks like this and sets up the RNN cells:

    def inference(self, key, context, sequences, num_enqueue_threads):
        # RNN cells and states
        cells = list()
        initial_states = dict()
        for i in range(0, self.num_layers):
            cell = tf.contrib.rnn.LSTMBlockCell(num_units=self.lstm_size) # Block LSTM version gives better performance #TODO Add linear projection option
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,input_keep_prob=1-self.input_dropout, output_keep_prob=1-self.output_dropout)
            cells.append(cell)
            initial_states["lstm_state_c_" + str(i)] = tf.zeros(cell.state_size[0], dtype=tf.float32)
            initial_states["lstm_state_h_" + str(i)] = tf.zeros(cell.state_size[1], dtype=tf.float32)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        [...]

It receives the key, context, and content of a sequence, and how many threads should be used to fetch the `RandomShuffleQueue` to provide input to the model. I found that `LSTMBlockCell` works slightly faster than the normal `LSTMCell` class. Now comes the neat bit. We can use the following code to let Tensorflow form batches from these sequences after splitting them up into chunks according to the unroll length. These batches also come with a context that ensures the hidden state is carried over from the last chunk of a sequence to the next, sparing us from a lot of hassle trying to implement and optimise that on our own.

    # BATCH INPUT
    self.batch = tf.contrib.training.batch_sequences_with_states(
        input_key=key,
        input_sequences=sequences,
        input_context=context,
        input_length=tf.cast(context["length"], tf.int32),
        initial_states=initial_states,
        num_unroll=self.num_unroll,
        batch_size=self.batch_size,
        num_threads=num_enqueue_threads,
        capacity=self.batch_size * num_enqueue_threads * 2)
    inputs = self.batch.sequences["inputs"]
    targets = self.batch.sequences["outputs"]

`inputs` and `targets` are part of the resulting batch and are formed during runtime. New sequences get pulled as soon as some in the batch are finished. In the following, the inputs are transformed from one-dimensional indices into one-hot vectors. Then, they are reshaped from an \[batch\_size,unroll\_length,vocab\_size\] tensor to a list of \[batch\_size,vocab\_size\] tensors with length unroll\_length, to conform with the RNN interface. Finally, we can use the state\_saving\_rnn with the state-saving batch we created beforehand, to get our outputs.

    # Convert input into one-hot representation (from single integers indicating character)
    print(self.vocab_size)
    embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, inputs)

    # Reshape inputs (and targets respectively) into list of length T (unrolling length), with each element being a Tensor of shape (batch_size, input_dimensionality)
    inputs_by_time = tf.split(1, self.num_unroll, inputs)
    inputs_by_time = [tf.squeeze(elem, squeeze_dims=1) for elem in inputs_by_time]
    targets_by_time = tf.split(1, self.num_unroll, targets)
    targets_by_time = [tf.squeeze(elem, squeeze_dims=1) for elem in targets_by_time] # num_unroll-list of (batch_size) tensors
    self.targets_by_time_packed = tf.pack(targets_by_time) # (num_unroll, batch_size)

    # Build RNN
    state_name = initial_states.keys()
    self.seq_lengths = self.batch.context["length"]
    (self.outputs, state) = tf.nn.state_saving_rnn(cell, inputs_by_time, state_saver=self.batch,
                                              sequence_length=self.seq_lengths, state_name=state_name, scope='SSRNN')

Here we have to be careful: If we have N characters, a special end-of-sequence token at the end of every sequence is appended while loading the data so the network learns when to stop generating lyrics, and I do not use zero as an index for any character as we need the zero entry to mask the output later to correctly compute the loss. So here, `self.vocab_size` would be N+2. Finally, we put a softmax on top of the outputs by iterating through the list of length num_unroll in which each entry represents one timestep, and return logits and probabilities:

    # Create softmax parameters, weights and bias, and apply to RNN outputs at each timestep
    with tf.variable_scope('softmax'):
        softmax_w = tf.get_variable("softmax_w", [self.lstm_size, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        logits = [tf.matmul(outputStep, softmax_w) + softmax_b for outputStep in self.outputs]

        self.logit = tf.pack(logits)

        self.probs = tf.nn.softmax(self.logit)
    tf.summary.histogram("probabilities", self.probs)
    return (self.logit, self.probs)

To train the model, we also need a loss function. Here we use the fact that we do not use the 0-index as a target, so we know that such an entry in the target list must come from the zero-padding and indicates that the sequence is already over. This is used in the following code to mask the loss by only considering non-zero targets. We also add L2 regularisation.

    def loss(self, l2_regularisation):
        with tf.name_scope('loss'):
            # Compute mean cross entropy loss for each output.
            self.cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logit, self.targets_by_time_packed) # (num_unroll, batchsize)

            # Mask losses of outputs for positions t which are outside the length of the respective sequence, so they are not used for backprop
            # Take signum => if target is non-zero (valid char), set mask to 1 (valid output), otherwise 0 (invalid output, no gradient/loss calculation)
            mask = tf.sign(tf.abs(tf.cast(self.targets_by_time_packed, dtype=tf.float32))) # Unroll*Batch \in {0,1}
            self.cross_loss = self.cross_loss * mask

            output_num = tf.reduce_sum(mask)
            sum_cross_loss = tf.reduce_sum(self.cross_loss)
            mean_cross_loss = sum_cross_loss / output_num # Mean loss is sum over masked losses for each output, divided by total number of valid outputs

            # L2
            vars = tf.trainable_variables()
            l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(l2_regularisation), weights_list=vars)

            loss = mean_cross_loss + l2_loss
            tf.summary.scalar('mean_batch_cross_entropy_loss', mean_cross_loss)
            tf.summary.scalar('mean_batch_loss', loss)
        return loss, mean_cross_loss, sum_cross_loss, output_num

Training the model
==================

Now that we set up the model, we need to train it!

Set up symbolic training operations
-----------------------------------

First we need the necessary symbolic operations. We set up a step counter and a learning rate variable that decays exponentially depending on the current step:

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0.0))
    # Learning rate
    initial_learning_rate = tf.constant(train_settings["learning_rate"])
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, train_settings["learning_rate_decay_epoch"], train_settings["learning_rate_decay_factor"])
    tf.summary.scalar("learning_rate", learning_rate

Then we calculate the gradients and prepare them for visualisation for Tensorboard, as you might run into the vanishing gradient problem for deeper RNNs:

    # Gradient calculation
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars, aggregation_method=2), # Use experimental aggregation to reduce memory usage
                                      5.0)

    # Visualise gradients
    vis_grads =  [0 if i == None else i for i in grads]
    for g in vis_grads:
        tf.summary.histogram("gradients_" + str(g), g)

We have to replace None entries with 0 for visualisation. We choose a gradient descent method (ADAM in this case) and define the training operations.

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.apply_gradients(zip(grads, tvars),
                                        global_step=global_step)

    trainOps = [loss, train_op,
              global_step, learning_rate]

Now we are ready to execute!

Performing training
-------------------

Start up a session and the QueueRunners associated with it. In our case, these are associated with the RNN input queue (NOT our own RandomShuffleQueue).

    # Start session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    tf_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

In case we crashed or want to import a (partly) trained model for other reasons such as fine-tuning, we check for previous model checkpoints to load the model parameters:

    # CHECKPOINTING
    #TODO save model directly after every epoch, so that we can safely refill the queues after loading a model (uniform sampling of dataset is still ensured)
    # Load pretrained model to continue training, if it exists
    latestCheckpoint = tf.train.latest_checkpoint(train_settings["checkpoint_dir"])
    if latestCheckpoint is not None:
          restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
          restorer.restore(sess, latestCheckpoint)
          print('Pre-trained model restored')

    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

Now we start a thread to run our custom `load_and_enqueue` method from earlier which reads from `data` and enqueues the sequences into the `RandomShuffleQueue`. Preprocessing and data loading is better done on the CPU.

    # Start a thread to enqueue data asynchronously to decouple data I/O from training
    with tf.device("/cpu:0"):
        t = threading.Thread(target=load_and_enqueue, args=[trainIndices])
        t.start()

We can set up some logging functions so we can nicely visualise statistics with Tensorboard:

    # LOGGING
    # Add histograms for trainable variables.
    histograms = [tf.summary.histogram(var.op.name, var) for var in tf.trainable_variables()]
    summary_op = tf.summary.merge_all()
    # Create summary writer
    summary_writer = tf.summary.FileWriter(train_settings["log_dir"], sess.graph.as_graph_def(add_shapes=True)

Now the training loop runs the training and summary operations, and writes summaries and periodically also model checkpoints to save progress:

    loops = 0
    while loops < train_settings["max_iterations"]:
        loops += 1
        [res_loss, _, res_global_step, res_learning_rate, summary] = sess.run(trainOps + [summary_op])
        new_time = time.time()
        print("Chars per second: " + str(float(model_settings["batch_size"] * model_settings["num_unroll"]) / (new_time - current_time)))
        current_time = new_time
        print("Loss: " + str(res_loss) + ", Learning rate: " + str(res_learning_rate) + ", Step: " + str(res_global_step))

        # Write summaries for this step
        summary_writer.add_summary(summary, global_step=int(res_global_step))
        if res_global_step % train_settings["save_model_epoch_frequency"] == 0:
            print("Saving model...")
            saver.save(sess, train_settings["checkpoint_path"], global_step=int(res_global_step))

After the maximum desired number of iterations has been reached (or some other criterion), we stop:
    
    # Stop our custom input thread
    print("Stopping custom input thread")
    sess.run(q.close())  # Then close the input queue
    t.join(timeout=1)

    # Close session, clear computational graph
    sess.close()
    tf.reset_default_graph()

Testing
=======

After training, we want to evaluate the performance on the test set. The code for this looks similar, so I will only show the differences here. We load the trained model:

    # CHECKPOINTING
    # Load pretrained model to test
    latestCheckpoint = tf.train.latest_checkpoint(train_settings["checkpoint_dir"])
    restorer = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    restorer.restore(sess, latestCheckpoint)

In our custom input queue thread, we close the queue after enqueueing all test samples so the process is stopped after seeing each example exactly once:

    # Enqueueing method in different thread, loading sequence examples and feeding into FIFO Queue
    def load_and_enqueue(indices):
        for index in indices:
            current_seq = data[index][1]
            sess.run(enqueue_op, feed_dict={keyInput: str(index),
                                          lengthInput: len(current_seq)-1,
                                        seqInput: current_seq[:-1],
                                        seqOutput: current_seq[1:]})
        print "Finished enqueueing all " + str(len(indices)) + " samples!"
        sess.run(q.close())

The test loop now uses the total cross-entropy returned from our model class along with the number of valid output positions to compute a bit-per-character metric:
    
    inferenceOps = [loss, mean_cross_loss, sum_cross_loss, output_num]

    current_time = time.time()
    logprob_sum = 0.0
    character_sum = 0
    iteration = 0
    while True:
        try:
            [l, mcl, scl, nb, summary] = sess.run(inferenceOps + [summary_op])
        except tf.errors.OutOfRangeError:
            print("Finished testing!")
            break

        new_time = time.time()
        print("Chars per second: " + str(
            float(model_settings["batch_size"] * model_settings["num_unroll"]) / (new_time - current_time)))
        current_time = new_time

        logprob_sum += scl # Add up per-char log probabilities of predictive model: Sum_i=1^N (log_2 q(x_i)), which is equal to cross-entropy term for all chars
        character_sum +=  nb # Add up how many characters were in the batch

        print(l, mcl, scl)
        summary_writer.add_summary(summary, global_step=int(iteration))
        iteration += 1

        print("Bit-per-character: " + str(logprob_sum / character_sum))

Note here that we catch an `OutOfRangeError` that is thrown as soon as our input queue is empty after the input thread finishes and closes it.

Sampling
========

Unfortunately, sampling as a use case is very different from the train and test setting:

*   We want to consider a single sequence, not a whole batch
*   The output of the RNN at the current time step is used as input for the next, which makes static unrolling inappropriate
*   We cannot in parallel evaluate multiple timesteps, but need to keep the hidden state after each input to feed back into the model (rendering the previously used state saver concepts cumbersome)

Therefore, I did it the hard way and maintained the RNN states myself. This requires setting up placeholders for the states manually to be able to feed values in during sampling, and defining the initial zero states to use for the prediction of the first character:

    # Load vocab
    vocab = Vocabulary.load(data_settings["input_vocab"])

    # INPUT PIPELINE
    input = tf.placeholder(tf.int32, shape=[None], name="input") # Integers representing characters
    # Create state placeholders - 2 for each lstm cell.
    state_placeholders = list()
    initial_states = list()
    for i in range(0,model_settings["num_layers"]):
        state_placeholders.append(tuple([tf.placeholder(tf.float32, shape=[1, model_settings["lstm_size"]], name="lstm_state_c_" + str(i)), # Batch size x State size
                                    tf.placeholder(tf.float32, shape=[1, model_settings["lstm_size"]], name="lstm_state_h_" + str(i))])) # Batch size x State size
        initial_states.append(tuple([np.zeros(shape=[1, model_settings["lstm_size"]], dtype=np.float32),
                              np.zeros(shape=[1, model_settings["lstm_size"]], dtype=np.float32)]))
    state_placeholders = tuple(state_placeholders)
    initial_states = tuple(initial_states)

The states are represented as tuples in tensorflow. The model itself also has to be adapted accordingly. We use a batch size and unroll length of 1, so we only predict exactly one character at a time, and feed in the input along with the state placeholders:

    # MODEL
    inference_settings = model_settings
    inference_settings["batch_size"] = 1 # Only sample from one example simultaneously
    inference_settings["num_unroll"] = 1 # Only sample one character at a time
    model = LyricsPredictor(inference_settings, vocab.size + 1)  # Include EOS token
    probs, state = model.sample(input, state_placeholders)

This time, we use the `sample` method from the `LyricsPredictor` class to build the required computational graph:

def sample(self, input, current_state):
    # RNN cells and states
    cells = list()
    for i in range(0, self.num_layers):
        cell = tf.contrib.rnn.LSTMBlockCell(num\_units=self.lstm\_size) # Block LSTM version gives better performance #TODO Add linear projection option
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,1.0,1.0) # No dropout during sampling
        cells.append(cell)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    self.initial\_states = cell.zero\_state(batch_size=1,dtype=tf.float32)

    # Convert input into one-hot representation (from single integers indicating character)
    embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
    input = tf.nn.embedding_lookup(embedding, input) # 1 x Vocab-size
    inputs\_by\_time = \[input\] # List of 1 x Vocab-size tensors (with just one tensor in it, because we just use sequence length 1

    self.outputs, state = tf.nn.rnn(cell, inputs\_by\_time, initial\_state=current\_state, scope='SSRNN')

Crucially, we set the scope when setting up the RNN to the same that was used when building the model during training and testing, so that when we load the checkpoint, the RNN variables are set up correctly. Afterwards, the softmax is applied as shown earlier. The function returns the probabilities and the resulting LSTM state after processing the input character, which we store in the following.

    inference = [probs, state]

    current_seq = "never" # This can be any alphanumeric text
    current_seq_ind = vocab.char2index(current_seq)

    # Warm up RNN with initial sequence
    s = initial_states
    for ind in current_seq_ind:
        # Create feed dict for states
        feed = dict()
        for i in range(0, model_settings["num_layers"]):
            for c in range(0, len(s[i])):
                feed[state_placeholders[i][c]] = s[i][c]
                feed[state_placeholders[i][c]] = s[i][c]

        feed[input] = [ind] # Add new input symbol to feed
        [p, s] = sess.run(inference, feed_dict=feed)

In the above code, we set an initial sequence ("never") and prepare the LSTM to continue the lyrics (e.g. "never gonna give you up") by feeding in one character after another and carrying over the states. These are nested tuples, organised according to layers, each with a cell and a hidden state (this is due to the LSTM structure). The hidden state now hopefully captures meaningful information about the input text `current_seq`, so we can take the current prediction probabilities and sample from them to generate the next character, feed it into the network, and repeat that process until we receive the special end token signalling that the LSTM is finished with the "creative process":

    # Sample until we receive an end-of-lyrics token
    iteration = 0
    while iteration < 100000: # Just a safety measure in case the model does not stop
        # Now p contains probability of upcoming char, as estimated by model, and s the last RNN state
        ind_sample = np.random.choice(range(0,vocab.size+1), p=np.squeeze(p))
        if ind_sample == vocab.size: # EOS token
            print("Model decided to stop generating!")
            break

        current_seq_ind.append(ind_sample)

        # Create feed dict for states
        feed = dict()
        for i in range(0, model_settings["num_layers"]):
            for c in range(0, len(s[i])):
                feed[state_placeholders[i][c]] = s[i][c]
                feed[state_placeholders[i][c]] = s[i][c]

        feed[input] = [ind_sample]  # Add new input symbol to feed
        [p, s] = sess.run(inference, feed_dict=feed)

        iteration += 1

Finally, we convert the generated list of integer indices to their character representation, and print out the result:

    c_sample = vocab.index2char(current_seq_ind)
    print("".join(c_sample))

    sess.close()
    
Example output
--------------

And now starts the fun! Feel free to extend the model to your liking. Want to see what my own model generates? Here is the output of a 2-layer 512-hidden node LSTM with 0.2 output dropout trained for only two hours on Metrolyrics text, when told to start with "never":

> never yet in you but that letters know a stobalal in you on the brink so to the victory no matter what i might understand the sun where i am with all this phon people theyll get my knife off a girl that it thats forsaken just smiling still welcome to me  
> its a gangsta good times is like a fesire then im holding fantastine is on though we bring it out to who burn today well make all the lights in his face im here so bright  
> sos we do what we do we dont know where we harm  
> and every time we get you now dont need rewith nothing yeah you dont want a sip or just look at dont make it on the 5 dirty doubt then most name yeah dont know about it i know and you dont wanna play with her no no no  
> come on ah yeah yeah women make you bury tight around rising in stop up in the top looking out of the middle of the sea  
> youre not drunk and im not in real hard to hit you all around cover up tune whats not there how to make you cry so long your cut money rolling around and the storm ignite it youre peacent burn so fast blue no fading  
> two number on the home was we praying of happy for your respect a death a lip another day and style niggas keep that an internuted at leven but you was the way you fall and ive been ready at all ive never seen the girls i tried to drive i took a fool from the river instead i just draw your life when my head stays the fellas we dreams to all ill stay  
> and nobody should be there in here but when i hide all my echo and make you hold me up im going to get more to fall in your sea oh right through a night in your news one two treatnessboy shes passed out in the sky all the real friends are downs light to out here he was rightly word out im not driving in my eyes suddenly reminding me im being that dragong class i wish i was since yours your peace of  
> pour it like like a record beautiful man i do you on the kid punk its when im attack i know when im smiling taste so i find a little far

As you can see, the model learned to structure its musings in paragraphs akin to real lyrics, and overall makes some good attempts at coming up with new sentences. Apparently this song is more of a Gangsta rap, as suggested by the words "knife", "gangsta", "niggas", etc. Sentences only sometimes make proper sense, unfortunately. It sometimes comes up with semi-random new words, like "dragong" and "peacent", because it has to learn spelling and vocabulary from scratch as opposed to word-level language models. It also did not learn meaningful long-term dependencies such as verse/chorus structures. 
