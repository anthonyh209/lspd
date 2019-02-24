"""LSPD algorithm"""

import os
import adler
adler.util.gpu.setup_one_gpu()

#from adler.odl.phantom import random_phantom
from adler.tensorflow import prelu, cosine_decay

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
import partial
from odl.discr import uniform_partition

np.random.seed(0)
name = os.path.splitext(os.path.basename(__file__))[0]

sess = tf.InteractiveSession()

print("Learned stochastic Relu CUDA. positive ellipses")
n_angles = 60 #number of projection angles

print("defining user selected parameters")
# User selected paramters
angles_in_batch = 10
if (n_angles % angles_in_batch != 0):
    print("number of angles must be divisible by the number of angles in the batch")
    exit()
else:
    n_batches=int(n_angles/angles_in_batch)

# Create ODL data structures
size = 128

space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

print("generating geometry")
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=n_angles)

print("generating operator")
# operator = odl.tomo.RayTransform(space, geometry)
operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda') #this operator to create the corresponding phantoms (with the full specified layer)

#operator is a function class (so simply calling the operator, will call out the function)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator


# Create tensorflow layer from odl operator
print("\ngenerating operator tensorflow layer")
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')

print("generating adjoint operator tensorflow layer")
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')

partial_op = partial.PartialRay(space, impl='astra_cuda')

print("preparing the partial layer")
odl_op_partial_layer = partial.tensor_partial_layer(partial_op, 'PartialRayTransform')
odl_op_partial_layer_adjoint = partial.tensor_partial_layer(partial_op.adjoint, 'PartialRayTransformAdjoint')


# Retrieving the angle projections
angle_partition = uniform_partition(0, np.pi, n_angles)
projections_array = (np.array(angle_partition.grid.coord_vectors)).ravel()
selection_array = np.arange(n_angles)

n_data = 5
n_iter = 10
n_primal = 5
n_dual = 5


def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    y_arr = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom = odl.phantom.shepp_logan(space, True)
        else:
            phantom = partial.random_phantom(space)
        data = operator(phantom)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

        x_true_arr[i, ..., 0] = phantom
        y_arr[i, ..., 0] = noisy_data

    return y_arr, x_true_arr



with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    original_y = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="original_y")
    y_rt = tf.placeholder(tf.float32, shape=[n_batches, None, angles_in_batch, operator.range.shape[1], 1], name="y_rt")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    epoch_angle = tf.placeholder(tf.float32, shape=[n_batches, angles_in_batch], name="epoch_projection_order")


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())



print("\ngenerating tomography ")
with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):


        primal = tf.concat([tf.zeros_like(x_true)] * n_primal, axis=-1)  # primal is always the full range
        dual = tf.concat([tf.zeros_like(y_rt[1])] * n_dual, axis=-1)

    for i in range(n_batches):  # iterations, the amount of projection batches we have
        with tf.variable_scope('dual_iterate_{}'.format(i)):

            evalop = odl_op_partial_layer(primal[..., 1:2], epoch_angle[i])
            update = tf.concat([dual, evalop, y_rt[i]], axis=-1)
            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_dual)
            dual = dual + update

        with tf.variable_scope('primal_iterate_{}'.format(i)):

            evalop = odl_op_partial_layer_adjoint(dual[..., 0:1], epoch_angle[i])
            update = tf.concat([primal, evalop], axis=-1)
            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_primal)
            primal = primal + update

    primal = tf.nn.relu(primal)
    x_result = primal[..., 0:1]



with tf.name_scope('loss'):
    residual = x_result - x_true
    squared_error = residual ** 2
    loss = tf.reduce_mean(squared_error)


with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 10001
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta2=0.99)

        tvars = tf.trainable_variables()
        tryout=tf.gradients(loss, tvars)
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('psnr', -10 * tf.log(loss) / tf.log(10.0))

    tf.summary.image('x_result', x_result)
    tf.summary.image('x_true', x_true)
    tf.summary.image('squared_error', squared_error)
    tf.summary.image('residual', residual)

    merged_summary = tf.summary.merge_all()
    test_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/test',
                                                sess.graph)
    train_summary_writer = tf.summary.FileWriter(adler.tensorflow.util.default_tensorboard_dir(name) + '/train')


print("\ninitialising Tensorflow variables")

# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
y_arr_validate, x_true_arr_validate = generate_data(validation=True)

validate_iteration_selection = selection_array
np.random.seed(4)
np.random.shuffle(validate_iteration_selection)
np.random.seed(None)
validate_iteration_selection = validate_iteration_selection.reshape(n_batches, angles_in_batch)
validate_iteration_selection = np.sort(validate_iteration_selection, axis=1)

print(validate_iteration_selection)
validate_epoch_angles = projections_array[validate_iteration_selection]

y_validate_set = np.empty([n_batches, y_arr_validate.shape[0], angles_in_batch, y_arr_validate.shape[2], y_arr_validate.shape[3]])
for j in range(n_batches):
    y_validate_set[j, :, :, :, :] = y_arr_validate[:, validate_iteration_selection[j], :, :]



if 1:
    saver.restore(sess, adler.tensorflow.util.default_checkpoint_path(name))

print("\nbegin training")
# Train the network
for i in range(0, maximum_steps):
    print("STEP:")
    print(i, "started")

    # Generating the angle sets for projections
    iteration_selection = selection_array
    np.random.shuffle(iteration_selection)
    iteration_selection = iteration_selection.reshape(n_batches, angles_in_batch)
    iteration_selection=np.sort(iteration_selection, axis=1)
    epoch_angles = projections_array[iteration_selection]

    # Example: a=(epoch_angles[1]) to get the first set of angles for the epoch iteration
    # a = (epoch_angles.shape)[0]

    # to be removed from this aspect (not into consideration for further purposes)

    #if i%10 == 0:
    y_arr, x_true_arr = generate_data()

    # preparing the data points inn batches (splitting the y values accordingly)
    y_values_set = np.empty([n_batches, y_arr.shape[0], angles_in_batch, y_arr.shape[2], y_arr.shape[3]])
    for j in range(n_batches):
        y_values_set[j, :, :, :, :] = y_arr[:, iteration_selection[j], :, :]

    _, merged_summary_result_train, global_step_result = sess.run([optimizer, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr,
                                         original_y: y_arr,
                                         y_rt: y_values_set,
                                         is_training: True,
                                         epoch_angle: epoch_angles})

    if i>0 and i%10 == 0: #loss, does not update the variables (it only calculates the loss)
        loss_result, merged_summary_result, global_step_result = sess.run([loss, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr_validate,
                                         y_rt: y_validate_set,
                                         original_y: y_arr,
                                         is_training: False,
                                         epoch_angle: validate_epoch_angles})


        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        test_summary_writer.add_summary(merged_summary_result, global_step_result)

        print('iter={}, loss={}'.format(global_step_result, loss_result))

    if i>0 and i%100 == 0:
        saver.save(sess, adler.tensorflow.util.default_checkpoint_path(name))



