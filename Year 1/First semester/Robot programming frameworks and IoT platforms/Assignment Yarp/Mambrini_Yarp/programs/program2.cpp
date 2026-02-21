#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>

#include <yarp/dev/all.h>
#include <yarp/sig/all.h>
#include <yarp/os/all.h>


using namespace std;
using namespace yarp::os;
using namespace yarp::dev;
using namespace yarp::sig;


int main(int argc, char *argv[]){

    Network yarp;
   
    // Lettura file dt
   
    
    double dt = 0.071;


    // --- Comunication control_world
    BufferedPort<Bottle> controlPort;

    controlPort.open("/program2/cmd_receiver");

    // --- Port for the image ---
    
    BufferedPort<ImageOf<PixelRgb>> imageport;

    imageport.open("/Client_Image");

    while(!yarp.connect("/icubSim/cam/left","/Client_Image")){
        Time::delay(1.0);
        cerr<<"I'm connecting for the image";
    }
    
    // Control head(like the other one)
    
    Property options;

    options.put("device","remote_controlboard");
    options.put("local","/program2/client");
    options.put("remote","/icubSim/head");

    PolyDriver p (options);

    if (!p.isValid()){
        cerr << "The driver is not working"<<endl;
        return 1;
    }


    IPositionControl * ip;
    IEncoders *en ;
    
    p.view(en);
    p.view(ip);

    bool move = true;
    bool tracking = false;



    // Variabili PID
    double P_iter; 
    double I_iter;
    double D_iter;
    
    vector<double> n_start = {50,50,35};
    
    vector<double> P_var = {0.7,0.7,0.85};             
    vector<double> I_var = {4, 5, 5.5};
    vector<double> D_var = {0.0104,0.0104,0.0103};
    double integralX = 0.0; double prevErrorX = 0.0;
    double integralY = 0.0; double prevErrorY = 0.0;
    
    
    
    vector<double> dataLog;
    int n_iteration = 0;
   
    double counter = 0;
    double c = 0;
    
    while(move){

        
        // Read command from the other program

        Bottle *cmd = controlPort.read(false);

        if(cmd != NULL){
            string msg_program1 = cmd->get(0).asString();
            cout << "Command : " << msg_program1 <<endl;
            if (msg_program1 == "Reset"){
                tracking = false;
                n_iteration = cmd->get(1).asInt32()-1;
               
                // Reset eye
                ip->positionMove(3,0.0);
                ip->positionMove(4,0.0);
                integralX = 0.0; prevErrorX = 0.0;
                integralY = 0.0; prevErrorY = 0.0;
                P_iter = 0.4;
                I_iter = 0;
                D_iter = D_var[n_iteration];
                c = 0; // For the PID
               
                
            }
            else if (msg_program1 == "Start") {
                tracking = true;
                
            }
            else if (msg_program1 == "Stop"){
                tracking = false;
                string filename = "dati_" + to_string(n_iteration+1) + ".csv";
                ofstream file(filename);
                
                // Transformation flat matrix
                for(size_t i=0; i<dataLog.size(); i+=6){
                    if(i+5 < dataLog.size()){
                        file << dataLog[i] << "," << dataLog[i+1] << "," 
                             << dataLog[i+2] << "," << dataLog[i+3] << "," 
                             << dataLog[i+4] << "," << dataLog[i+5] << endl;
                    }
                }
                file.close();
                cout << "File saved: " << filename << endl;
                dataLog.clear();
               
            }
            else if (msg_program1 == "Quit"){
                move = false;
                
            }
          
        }
    // Find the centre of the sphere
        if (tracking && move){
            
            c++;
            ImageOf<PixelRgb> *image = imageport.read(false);
        
            double xMean = 0;
            double yMean = 0;
            int counter = 0 ;

            if (image!=NULL){
            
                for (int y = 0; y<image->height();y++){               
                    for (int x = 0; x<image->width(); x++){                
                    PixelRgb & pixel = image->pixel(x,y);
                    if (pixel.r >pixel.b*1.2+10 && pixel.r>pixel.g*1.2+10){ 
                    xMean += x;
                    yMean += y;
                    counter++;
                    }
                }            
            }
            }
   
        if (counter != 0){
            double center_x = (xMean)/counter;
            double center_y = (yMean)/counter;
            

            double joint_0 = 0; double joint_3 = 0;
            double joint_2 = 0; double joint_4 = 0;
        
            en->getEncoder(0,&joint_0); en->getEncoder(3,&joint_3);
            en->getEncoder(2,&joint_2); en->getEncoder(4,&joint_4);

            // Find the error

            double errorX = (160-center_x); 
            double errorY = (120-center_y); 

            integralX += errorX*dt; integralY += errorY*dt;
            double derivativeX = (errorX - prevErrorX)/dt;
            double derivativeY = (errorY - prevErrorY)/dt;
            
            if (c>n_start[n_iteration]){
                I_iter = I_var[n_iteration];
                P_iter = P_var[n_iteration]; 
                
            }
            double x_value = joint_4-(P_iter*errorX+I_iter*integralX+D_iter*derivativeX);
            double y_value = P_iter*errorY+joint_3+I_iter*integralY+D_iter*derivativeY;

            prevErrorX = errorX;
            prevErrorY = errorY;
            
            // Make the move

            ip->positionMove(4,x_value);
            ip->positionMove(3,y_value);

            dataLog.push_back(errorX); dataLog.push_back(errorY); 
            dataLog.push_back(joint_0); dataLog.push_back(joint_2); 
            dataLog.push_back(joint_3);dataLog.push_back(joint_4);
        }
    }
    }

    p.close();
    imageport.close();
    controlPort.close();
    return 0;
}
