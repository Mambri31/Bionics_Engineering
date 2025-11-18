import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from action_tutorials_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            node=self,
            action_type=Fibonacci,
            action_name='fibonacci',
            execute_callback=self.execute_callback)
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        fbck_msg = Fibonacci.Feedback(partial_sequence=[0, 1])
        for i in range(1, goal_handle.request.order):
            fbck_msg.partial_sequence.append(
                fbck_msg.partial_sequence[i] + fbck_msg.partial_sequence[i-1])
            self.get_logger().info('Feedback: {fbck_msg.partial_sequence}')
            goal_handle.publish_feedback(fbck_msg)
            time.sleep(1)
        goal_handle.succeed()
        return Fibonacci.Result(sequence=fbck_msg.partial_sequence)






def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)

if __name__ == '__main__':
    main()
