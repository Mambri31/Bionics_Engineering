#include <yarp/os/Bottle.h>
#include <iostream>
#include <time.h>
#include <yarp/os/all.h>

 using namespace yarp::os;
 using namespace std;

int main(int argc, char *argv[]){
    Network yarp;
    Port clientPort;
    clientPort.open("/myClient");
    bool check = yarp.connect("/myClient", "/icubSim/head/rpc:i");
    if(!check)
    {
        cerr << "Failed to connect to the server"<<endl;
        return 1;
    }
    Bottle req;
    for(int i = 0; i<6;i++)
    {
        for(int step = -30; step < 30;step++)
        {
            req.addString("set");
            req.addString("pos");
            req.addInt8(i);
            req.addInt8(step);
            clientPort.write(req);
        }
        Time::delay(2); 
    }
    clientPort.close();
    return 0;

}
