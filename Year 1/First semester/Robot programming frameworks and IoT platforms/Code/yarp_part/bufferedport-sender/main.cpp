#include <yarp/sig/Vector.h>
#include <yarp/os/Bottle.h>
#include <iostream>
#include <time.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>

 using namespace yarp::sig;
 using namespace yarp::os;
 using namespace std;

 int main(int argc, char *argv[]){
    srand(time(NULL));
    Network yarp;
    BufferedPort<VectorOf<int>> bp;
    bool ok = bp.open("/sender");
    if(!ok){
        cerr << "Failed to open port"<<endl;
        return 1;
    }

    yarp.waitConnection("/sender", "/receiver");
    VectorOf<int> & v=bp.prepare();
    
    v.resize(20);

    for(int i = 0; i < v.size(); i++)
    {
        v[i] = rand()%30 +1;
    }

    bp.write();
    //bp.close();
    return 0;
 }