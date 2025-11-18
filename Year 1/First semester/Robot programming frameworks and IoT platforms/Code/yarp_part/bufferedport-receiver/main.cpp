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
    Network yarp;
    BufferedPort<VectorOf<int>> bp;
    bool ok = bp.open("/receiver");

    if(!ok){
        cerr << "Failed to open port"<<endl;
        return 1;
    }

    yarp.connect("/sender","/receiver");
    VectorOf<int> * v = bp.read();

    cout <<"Vector stamp: ";
    for(int i = 0; i < v->length(); i++)
        cout<<" "<< (*v)[i];
    //DANIELE's solution
    int medianPos = 0;
    int i,j = 0;

    double median = 0.0;
    int length = v->length();
    VectorOf<int> countVect;
    countVect.resize(length);

    countVect.zero();

    if(length%2==1)
    {
        for(int i =0; i < length; i++)
        {
            for(j = 0; j < length; j++)
            {
                if((*v)[i]< (*v)[j])
                    countVect[i]++;
            }// end j for
        }// end i for
        for(i=0; i < length; i++)
        {
            if(countVect[i]<countVect[medianPos])
                medianPos = i;

        }

        for(i=0; i < length; i++)
        {
            if(countVect[i]<=length/2 && countVect[i]>countVect[medianPos])
                medianPos = i;
        }

        median = (*v)[medianPos];
        cout<<"Median: "<<median;
    }
    else
    {
        bool skip1 = false;
        bool skip2 = false;
        int value1 = 0;
        int value2 = 0;
        double median1 = 0.0;
        for(int i = 0; i < length; i++)
        {
            int value = (*v)[i];
            int lesser = 0;
            int equal = 0;
        
            for(int j = 0; j < length; j++)
            {
                if((*v)[j]<value)
                    lesser++;
                else if ((*v)[j]==value)
                    equal ++;
                
            }
            if(length/2<=(lesser + equal) && (length/2)>lesser)
            {
                value1 = (*v)[i];
                skip1 = true;
            }
            if(((length/2)+1)<=(lesser + equal) && ((length/2)+1)>lesser)
            {
                value2 = (*v)[i];
                skip2 = true;
            }
            if(skip1&&skip2)
            {
                median1 = (value1 + value2)/2.0;
                cout<<"Median: "<<median1;
                break;
            }

        }
    }


    
    return 0;
 }