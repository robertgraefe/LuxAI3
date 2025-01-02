import tensorflow as tf
from env import LuxAIS3PyEnv

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec, BoundedArraySpec
from tf_agents.utils import common
import tf_agents
import numpy as np


# https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb

class Agents:
    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    env: LuxAIS3PyEnv

    def __init__(self):

        env = LuxAIS3PyEnv()

        action_spec = (10, 10)
        action_tensor_spec = BoundedArraySpec(shape=(), dtype=np.float32, minimum=0, maximum=10, name='action')
        #action_tensor_spec = tf.TensorSpec(action_spec)
        num_actions = 10

        fc_layer_params: tuple[int, int] = (100, 50)
        dense_layers = [self.dense_layer(num_units) for num_units in fc_layer_params]

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.

        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        train_step_counter = tf.Variable(0)

        time_step_spec = tf_agents.trajectories.time_step_spec()
        agent = dqn_agent.DqnAgent(
            time_step_spec,
            action_tensor_spec,
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        agent.initialize()

        eval_policy = agent.policy
        collect_policy = agent.collect_policy
        random_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_tensor_spec)

        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(
            agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)

        table = reverb.Table(
            table_name,
            max_size=self.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        replay_buffer = tf_agents.replay_buffers.reverb_replay_buffer.ReverbReplayBuffer(agent.collect_data_spec, table_name, sequence_length=2, local_server=reverb_server)

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2)

        py_driver.PyDriver(
            env,
            py_tf_eager_policy.PyTFEagerPolicy(
                random_policy, use_tf_function=True),
            [rb_observer],
            max_steps=self.initial_collect_steps).run(env.reset())

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(self, num_units: int):

        input = tf.keras.layers.InputLayer(tf.keras.Input((10,10)))

        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))([input])

    def compute_avg_return(self, policy, num_episodes=10) -> float:

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = self.env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = self.env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        #return avg_return.numpy()[0]#
        return avg_return

##

agent = Agents()
print(agent.env.reset())