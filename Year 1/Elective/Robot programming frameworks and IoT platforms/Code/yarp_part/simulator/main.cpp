#include <yarp/os/all.h>
#include <yarp/dev/all.h>
#include <iostream>


using namespace yarp::os;
using namespace yarp::dev;
using namespace std;

int main(int argc, char *argv[]) {


// Set up YARP
Network yarp;

Property options;
options.put("device","remote_controlboard");
options.put("local","/simulator/client");
options.put("remote","/icubSim/head");

PolyDriver p (options);

IPositionControl * ip;
IEncoders * en;

p.view(ip);
p.view(en);

double * enco = new double [6];

for (int ii = 0; ii<6; ii++){
    for (int jj = 0; jj<30; jj++){
        ip->positionMove(ii,jj);
        cout<<"\nJoint "<<ii<<" moved at position :"<<jj<<endl;
        en->getEncoders(enco);
        cout<<"Encoder values: ";
        for(int i = 0; i < 6; i++)
            cout<<"\t"<<enco[i];

    }
    //cout<<"\n";
}



return 0;



}
