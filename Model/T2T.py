import numpy as np
import tensorflow as tf
class T2T_Model():
    def __init__(self, vocab_size, num_units, dropout_rate, num_block, is_training
                , num_heads, hidden_units):
         self.vocab_size = vocab_size
         self.num_units = num_units
         self.dropout_rate = dropout_rate
         self.num_block = num_block
         self.is_training = is_training
         self.num_heads = num_heads
         self.hidden_units = hidden_units

    def encoder(self, inputs, emb = True, causality = False, zero_pad = True, scale = True, scope = "encoder", reuse = None):
        with tf.variable_scope(scope, reuse = reuse):
            if emb:
                self.word_emb = self.embedding(inputs, self.vocab_size, self.num_units)
                self.pos_emb = self.positional_encoding(inputs, self.num_units)
                self.enc = self.word_emb + self.pos_emb
            else:
                self.enc = inputs
            self.enc = tf.layers.dropout(self.enc, rate=self.dropout_rate, training = self.is_training)
            self.enc_layers = []
            for i in range(self.num_block):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    self.enc, _ = self.multihead_attention(queries=self.enc,
                                                        keys=self.enc,
                                                        num_units= self.num_units,
                                                        num_heads= self.num_heads,
                                                        dropout_rate= self.dropout_rate,
                                                        is_training= self.is_training,
                                                        causality= causality,
                                                        scope = "enc_self_attention_block_" + str(i))
                    self.enc, self.enc_layer = self.feedforward(self.enc, num_units=[4*self.hidden_units, self.hidden_units], scope = "enc_feedward_block_" + str(i))
                    #self.output = tf.reshape(tf.concat(self.enc_out, 1), [-1, self.out_size])
                    #self.enc_layers = tf.stack([self.enc_layers, self.output], axis = 0)
                    #self.enc = tf.Print(self.enc, [self.enc], summarize = 20)
            return self.enc, self.enc_layers
    
    def Trans_Pointer(self, inputs, outputs, emb = True, causality= False, zero_pad = True, scale = True, reuse = None):
        self.enc, _ = self.encoder(inputs, emb = emb, causality = True)
        self.dec, pointer = self.decoder(outputs, self.enc, emb = emb)
        return self.dec, pointer

    def encode(self, inputs, causality= False, zero_pad = True, scale = True, reuse = None):
        encode_ans = self.encoder(inputs, causality)
        enc = tf.reduce_sum(encode_ans, axis = 1)
        enc = tf.reshape(enc, [-1, self.num_units])
        enc = tf.layers.dense(enc, self.hidden_units, activation=tf.nn.softsign)
        return enc

    def normalize(self,
                      inputs,
                      epsilon = 1e-8,
                      scope="ln",
                      reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                inputs_shape = inputs.get_shape()
                params_shape = inputs_shape[-1:]

                mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
                beta= tf.Variable(tf.zeros(params_shape))
                gamma = tf.Variable(tf.ones(params_shape))
                normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
                outputs = gamma * normalized + beta

            return outputs

    def decoder(self,
                inputs,
                encoder,
                emb = True,
                dec_causality = False,
                zero_pad = True,
                scale = True,
                scope = "decoder",
                reuse = None):
        with tf.variable_scope(scope, reuse=reuse):
            if self.num_units is None:
                self.num_units = queries.get_shape().as_list[-1]
            if emb:
                self.dec_word_emb = self.embedding(inputs, self.vocab_size, self.num_units)
                self.dec_pos_emb = self.positional_encoding(inputs, self.num_units)
                self.dec = self.dec_word_emb + self.dec_pos_emb
                self.dec = tf.layers.dropout(self.dec,
                                             rate=self.dropout_rate,
                                             training = self.is_training)
            else:
                self.dec = inputs
            for i in range(self.num_block):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    self.dec, _ = self.multihead_attention(queries=self.dec,
                                                        keys=self.dec,
                                                        num_units= self.num_units,
                                                        num_heads= self.num_heads,
                                                        dropout_rate= self.dropout_rate,
                                                        is_training= self.is_training,
                                                        causality= dec_causality,
                                                        scope = "dec_mask_attention"
                                                        )
                    
                    self.dec, pointer = self.multihead_attention(queries = self.dec,
                                                        keys=encoder,
                                                        num_units= self.num_units,
                                                        num_heads= self.num_heads,
                                                        dropout_rate= self.dropout_rate,
                                                        is_training= self.is_training,
                                                        causality=False,
                                                        scope = "dec_self_attention"
                                                        )
                    self.dec, _ = self.feedforward(self.dec, num_units=[4*self.num_units, self.num_units])

        return self.dec, pointer

    def seq2seq(self,
                inputs,
                outputs
               ):
        self.enc = self.encoder(inputs, causality = True)
        self.dec, pointer = self.decoder(outputs, self.enc)
        self.dec = tf.layers.dense(self.dec, self.hidden_units, tf.nn.relu)

        return self.dec, pointer

    def embedding(self,
                      inputs,
                      vocab_size,
                      num_units,
                      zero_pad=True,
                      scale=True,
                      scope="embedding",
                      reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                lookup_table = tf.get_variable('lookup_table',
                                                dtype=tf.float32,
                                                shape=[vocab_size, num_units],
                                                initializer=tf.contrib.layers.xavier_initializer())
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                           lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)
            #outputs = tf.Print(outputs, [outputs], summarize = 80)
            if scale:
                outputs = outputs * (num_units ** 0.5)

            return outputs

    def positional_encoding(self,
                                inputs,
                                num_units,
                                zero_pad=False,
                                scale=True,
                                scope="positional_encoding",
                                reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                N, T = inputs.get_shape().as_list()
                position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])

            # First part of the PE function: sin and cos argument
                position_enc = np.array([
                                 [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
                                  for pos in range(T)])
            # Second part, apply the cosine to even columns and sin to odds.
                position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
                position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
                lookup_table = tf.convert_to_tensor(position_enc, dtype = tf.float32)

                if zero_pad:
                    lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                                  lookup_table[1:, :]), 0)
                outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

                if scale:
                    outputs = outputs * num_units**0.5

            return outputs



    def multihead_attention(self,
                                queries,
                                keys,
                                num_units=None,
                                num_heads=8,
                                dropout_rate=0,
                                is_training=True,
                                causality=False,
                                scope="multihead_attention",
                                query_mask = False,
                                key_mask = False,
                                reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
                if num_units is None:
                    num_units = queries.get_shape().as_list[-1]

            # Linear projections
                #print "Q"
                #print queries.get_shape().as_list()
                #print num_units
                Q = tf.layers.dense(queries, num_units, activation = tf.nn.relu) # (N, T_q, C)
                K = tf.layers.dense(keys, num_units, activation = tf.nn.relu) # (N, T_k, C)
                V = tf.layers.dense(keys, num_units, activation = tf.nn.relu) # (N, T_k, C)

            # Split and concat
                Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
                Q_ = tf.Print(Q_, [Q_], summarize = 20, message = scope + " Q_")
                K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
                K_ = tf.Print(K_, [K_], summarize = 20, message = scope + " K_")
                V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

            # Multiplication
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
                outputs = tf.Print(outputs, [outputs], summarize = 20, message = scope + " output")

            # Scale
                outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
                outputs = tf.Print(outputs, [outputs], summarize = 20, message = scope + " output")
                 
            # Key Masking
                if key_mask:
                    key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
                    #key_masks = tf.Print(key_masks, [key_masks], summarize = 20)
                    key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
                    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
                    key_masks = tf.Print(key_masks, [key_masks], summarize = 20, message = scope + " key_mask")
                    #key_masks = tf.Print(key_masks, [key_masks], summarize = 20)
                    paddings = tf.ones_like(outputs)*(-2**32+1)
                    outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
                    outputs = tf.Print(outputs, [outputs], summarize = 20, message = scope + " output")

            # Causality = Future blinding
                if causality:
                    diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
                    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
                    paddings = tf.ones_like(masks)*(-2**32+1)
                    outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
                    #outputs = tf.Print(outputs, [outputs], summarize = 20)
            
            # Query Masking
                if query_mask:
                    query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
                    #key_masks = tf.Print(key_masks, [key_masks], summarize = 20)
                    query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
                    query_masks = tf.tile(tf.expand_dims(query_masks, 2), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
                    #query_masks = tf.Print(query_masks, [query_masks], summarize = 100)
                    #key_masks = tf.Print(key_masks, [key_masks], summarize = 20)
                    paddings = tf.ones_like(outputs)*(-2**32+1)
                    outputs = tf.where(tf.equal(query_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
                    outputs = tf.Print(outputs, [outputs], summarize = 20, message = scope + " output")
          
                #key_masks = tf.Print(key_masks, [key_masks], summarize = 20)
            # Activation
                attent_vec = tf.stack(tf.split(outputs, num_heads, axis=0))
                attent_vec = tf.reduce_max(attent_vec, axis = 0)
                attent_vec = tf.layers.dropout(attent_vec, rate=dropout_rate, training=is_training)
                #print attent_vec.get_shape().as_list()
                outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
                #attent_vec = tf.split(attent_vec, num_heads, axis=0)[0] # (N, T_q, C)

            # Query Masking
                #query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
                #query_masks = tf.Print(query_masks, [query_masks], summarize = 20)
                #query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
                #query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
                #query_masks = tf.Print(query_masks, [query_masks], summarize = 20)
                #outputs *= query_masks # broadcasting. (N, T_q, C)
                #query_masks = tf.Print(query_masks, [query_masks], summarize = 20)
                #outputs = tf.Print(outputs, [outputs], summarize = 20)
           
            # Dropouts
                outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

            # Weighted sum
                outputs = tf.matmul(outputs, V_) # (h*N, T_q, C/h)

            # Restore shape
                outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # (N, T_q, C)

            # Residual connection
                outputs += queries

            # Normalize
                outputs = self.normalize(outputs) # (N, T_q, C)

            return outputs, attent_vec

    def feedforward(self,
                        inputs,
                        form = "linear",
                        num_units=[2048, 512],
                        scope="multihead_attention",
                        reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                if form == "linear":
                    # Linear
                    outputs = tf.layers.dense(inputs, num_units[1], tf.nn.relu)
                    #Layer Output
                    outputs_layer = outputs
                    outputs_layer = self.normalize(outputs_layer)
                    # Residual connection
                    outputs += inputs
                    # Normalize
                    outputs = self.normalize(outputs)

                elif form == "cnn":
                    # Inner layer
                    params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                               "activation": tf.nn.relu, "use_bias": True}
                    outputs = tf.layers.conv1d(**params)

                    # Readout layer
                    params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                              "activation": None, "use_bias": True}
                    outputs = tf.layers.conv1d(**params)

                    # Residual connection
                    outputs += inputs

                    # Normalize
                    outputs = self.normalize(outputs)

            return outputs, outputs_layer

    def label_smoothing(self, inputs, epsilon=0.1):
            K = inputs.get_shape().as_list()[-1] # number of channels
            return ((1-epsilon) * inputs) + (epsilon / K)
