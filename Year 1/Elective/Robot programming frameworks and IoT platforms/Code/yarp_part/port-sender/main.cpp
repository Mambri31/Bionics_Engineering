#include <yarp/os/all.h>

#include <iostream>


using namespace yarp::os;



int main(int argc, char *argv[]) {


// Set up YARP

Network yarp;


// Create port

Port p;

bool ok = p.open("/examples/port/sender");

if (!ok) {

std::cerr << "Failed to open port" << std::endl;

return 1;

}


// waiting for the receiver before transmitting

yarp.waitConnection("/examples/port/sender", "/examples/port/receiver");


// send data in a loop



// prepare the message using a bottle

Bottle b;

int a=0;

srand (time(NULL));

for (int ii=0;ii<10;ii++){

b.clear();

a=rand()%3+1;


if (a==1){

b.addInt8(3);

b.addInt8(8);


}

else if (a==2) {

b.addString("hello");


}

else{


b.addString ("ciao");


}


// send the message through the port

//std::cout << "Sending Hello message " << std::endl;

p.write(b);


}


// close port

p.close();


return 0;



}
