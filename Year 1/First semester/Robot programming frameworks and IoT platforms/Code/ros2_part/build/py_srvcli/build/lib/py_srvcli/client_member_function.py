import sys
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__("minimal_client_async")

        self.cli = self.create_client(AddTwoInts, "add_two_ints")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        self.req = AddTwoInts.Request()

    def send_request(self,a,b):
        self.req.a = a
        self.req.b = b
        return self.cli.call_async(self.req)


def main():
    rclpy.init()
    #Template for a single short calling
    minimal_client = MinimalClientAsync()
    future = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    rclpy.spin_until_future_complete(minimal_client, future)
    response = future.result()
    #usiamo un meccanismo asincrono con una variabile chiamata future. Finchè la variabile future (che è il risultato della funzione di request) non avrà un return, noi spinniamo il nodo. 
    # quando la variabile future ha espletato il suo scopo, allora salviamo tale valore nella variabile response

    minimal_client.get_logger().info(f'Result of add_two_ints: for {int(sys.argv[1])} + {int(sys.argv[2])} = {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
