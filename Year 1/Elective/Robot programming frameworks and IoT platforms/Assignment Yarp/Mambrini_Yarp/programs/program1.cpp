#include <iostream>
#include <vector>
#include <cmath>
#include <yarp/os/all.h>
#include <yarp/dev/all.h>

using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
// He didnt complain about the code, he said that its well written
int main(int argc, char *argv[]){
    
    Network yarp;

    // Connection between the two programs
    
    BufferedPort<Bottle> port_wr;

    port_wr.open("/program1/cmd_sender");

    while (!yarp.connect("/program1/cmd_sender","/program2/cmd_receiver")){
        Time::delay(1.0);
        cout<< "Trying connection between the two programs"<<endl;
    }

    // Connection to the world 

    RpcClient clientPort;
    
    clientPort.open("/myClient");

    bool check = yarp.connect("/myClient", "/icubSim/world");
    
    if(!check){  
        cerr << "Failed to connect to the World"<<endl;
        return 1;
    }

    // Generation sphere 

    Bottle sphere, reply ;

    float radius = 0.04;
    float x = 0.0 ; float y = 0.9 ; float z = 0.8 ;

    float r = 1 ; float g = 0 ; float b = 0 ;


    sphere.addString("world"); sphere.addString("mk"); sphere.addString("ssph");
    sphere.addFloat64(radius);
    sphere.addFloat64(x); sphere.addFloat64(y); sphere.addFloat64(z);
    sphere.addFloat64(r); sphere.addFloat64(g); sphere.addFloat64(b);


    if(!clientPort.write(sphere,reply)){
        cout<<"Creation sphere failed"<<endl;
        return 1;
    }


    clientPort.close();

    // Control Head 

    Property options;

    options.put("device","remote_controlboard");
    options.put("local","/program1/client");
    options.put("remote","/icubSim/head");

    PolyDriver p_head (options);

    if (!p_head.isValid()){
        cerr << "Head driver is not working"<<endl;
        return 1;
    }

    IPositionControl * ip_head;
    
    p_head.view(ip_head);
    
    vector<double> f = {0.1,0.2,0.4};

    double duration = 15.0;

    // Change iteration 

    for (int i = 0; i<3 ; i++){
        
        cout<< "====== Starting iteration " << i+1 <<" (freq : "<<f[i] <<"Hz) ======" <<endl;

        // Reset eye

        Bottle& cmdReset = port_wr.prepare();
        cmdReset.clear();
        cmdReset.addString("Reset"); cmdReset.addInt32(i+1);

        port_wr.write();

        cout << "Waiting 5 seconds " << endl;
        Time::delay(5.0);
        Bottle & cmdStart =port_wr.prepare();
        cmdStart.clear();
        cmdStart.addString("Start");
        port_wr.write();

        // Movement Head 
        double t0 = Time::now(); // Inizio dei 15 s
    
        while(true){
        
        
            double t = Time::now()-t0;
        
            if (t>duration)
            break;
        
            double angle = 16*sin(2*M_PI*f[i]*t); // Sinusoide
        
            ip_head->positionMove(0,angle);
            ip_head->positionMove(2,angle);
     
        }
        Bottle& cmdStop = port_wr.prepare();
        cmdStop.clear();
        cmdStop.addString("Stop");
        port_wr.write();
        Time::delay(2);
    }

    // Ending 
    Time::delay(5.0);
    
    
    Bottle& cmdQuit = port_wr.prepare();
    cmdQuit.clear();
    cmdQuit.addString("Quit");
    port_wr.write();
    Time::delay(2);

    p_head.close();
    return 0;
}
