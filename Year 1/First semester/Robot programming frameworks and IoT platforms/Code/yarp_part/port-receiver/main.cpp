#include <yarp/os/all.h>
#include <iostream>

using namespace yarp::os;
using namespace std;

int main(){

    Network yarp;


    Port p;
    if (!p.open("/examples/port/receiver")) {
        cerr << "Failed to open receiver port" << endl;
        return 1;
    }

    Bottle b;
    Value v;
    
    int sum = 0;
    yarp.connect("/examples/port/sender","/examples/port/receiver");
    // Allora la bottle è tipo un array di celle indipendenti
    // con b.get(...) returna un oggetto Value(che è un'altro dataType) con
    // .asInt8() se nel value hai un integer returni l'integer
    for (int j=0; j<10; j++){
        
        p.read(b); // reading from the port
        v = b.get(0);

        if (v.isInt8()){
           sum = v.asInt8()+b.get(1).asInt8();

           cout << "Message type 1. The sum is : "<< sum <<"\n";

        }
        else {
            string s = v.asString();
            int count;
            for (int ii = 0; s[ii]!='\0';ii++){
               count ++;
            }
            if (count %2 == 0){
                cout << "Message type 2. Printing half of the string: ";
                for (int hh = 0;hh<= (count/2); hh++){
                    
                    if (hh == (count/2)){
                        cout << s[hh] << "\n";
                        break;
                    }
                    cout << s[hh];
                }
            
            }
            else {
                cout << "Message type 3. Printing full string: "<< s<<"\n";
            }
        }
        Time::delay(2); 
    }
    p.close();
    return 0;




}