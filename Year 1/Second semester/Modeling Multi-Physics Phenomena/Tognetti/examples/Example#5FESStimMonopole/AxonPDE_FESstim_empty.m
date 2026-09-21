%Propagation of the action potential (PDA) in an unmyelinated axon

clear variables 
close all
clc




%% problem data

L = 5; % axon length (cm)
t_sim=150; % simulation time in msec


Cm = 1; % uF/cmˆ2
GNa_ = 120; % mS/cmˆ2 
GK_ = 36; % mS/cmˆ2
GL = 0.300; % mS/cmˆ2
sigma_i = 1.6667; % mS/cm    
a = 0.005; % cm
Vr=-60; %membrane rest potential (mV)
VNa=55; %absolute value of Na reversal potential  (mV)
VK=72;  %absolute value of K reversal potential  (mV)
VL=49.387; %Leakage potential  (mV)



% stimulation 
Is=8e3; % [uA]
d=0.6; %[cm]
s_stim=1%0.05 % current stimulation site on the axon (cm)
t_on=2; %current stimulus on (msec)
t_high=1; %duration of the stimulus (msec)
t_off=t_on+t_high; %current stimulus off (msec)
%%%%%%


ds = 0.01; % s_increment (cm)
H = round(L/ds); % total spatial nodal points
js=round(s_stim/ds);% spatial nodal index of stimulation
s=(0:ds:L);

dt= 0.01; %t_increments msec (try multyply by 2.5 to obtain a non stable condition
N=round(t_sim/dt); %total temporal nodal points
t=(0:dt:t_sim);
i_on=round(t_on/dt); %temporal nodal index of t_on
i_off=round(t_off/dt); %temporal nodal index of t_off




%% initial conditions on Vm, n, m, h

Vm = ones(N+1,H+1)*Vr; 

m0=0.0529;  %gating variables are dimensionless
h0=0.5961;
n0=0.3177;

%note: initial values of gating variables can be calculated with the
%following formulas: 

%Vm-Vr=0 then:
%m0=am(0)/(am(0)+bm(0)); %0.0529
%h0=ah(0)/(ah(0)+bh(0)); %0.5961
%n0=an(0)/(an(0)+bn(0)); %0.3177

m = ones(N+1,H+1)*m0;
h = ones(N+1,H+1)*h0;
n = ones(N+1,H+1)*n0;

sigma_o=1;% mS/cm  

% activation function F

figure
plot(s,Fact);




for i=1:N %time
    for j=2:H %s -> j=1 and j=H+1 -> zero flux boundary conditions
    
        %Explicit method (iterative): (starting from the initial
        %conditions), at time i we calculate the unknwons at time i*deltat+1
        

        %% calculus of Vm at next time i+1

        GNa=GNa_*(m(i,j)^3)*h(i,j);
        GK=GK_*(n(i,j)^4);
        INa=GNa*(Vm(i,j)-VNa); %Na
        IK=GK*(Vm(i,j)+VK); %K
        IL=GL*(Vm(i,j)+VL); %L
        Iion=IK+INa+IL; % cIion 
        
        %explicit calculation of Vm
        
      
     

        %% calculus of the gating variables (m,n,h) at next time i+1 
                
        dv=Vm(i,j)-Vr;
        an=( (0.01*(10-(dv)))/(exp((10-dv)/10)-1) ); % msec^-1
        bn=( 0.125/(exp(dv/80)) ); % msec^-1
        am=( (0.1*(25-(dv)))/(exp((25-dv)/10)-1) ); % msec^-1
        bm=( 4/(exp(dv/18)) ); % msec^-1
        ah=( 0.07/(exp(dv/20)) ); % msec^-1
        bh=( 1/(exp((30-dv)/10)+1) ); % msec^-1
                 
        m(i+1,j)= m(i,j)+dt*( am*(1-m(i,j))-bm*m(i,j) );
        h(i+1,j)= h(i,j)+dt*( ah*(1-h(i,j))-bh*h(i,j) );
        n(i+1,j)= n(i,j)+dt*(  an*(1-n(i,j))-bn*n(i,j) );
             
    end
        %zero flux boundary
        Vm(i+1,1)=Vm(i+1,2);
        Vm(i+1,H+1)=Vm(i+1,H);
end


    






%% animate results

    nframe=200;
    DT=round((N-1)/nframe); % to plot every nframe frames
    figure
    a1=animatedline('Color','b','LineWidth',2);
    axis([s(1) s(end) -90 50]); 
    ylabel('[mV]');
    xlabel('[cm]');
    legend('Vm');
    for i=1:DT:N
        clearpoints(a1);
        addpoints(a1,s,Vm(i,:));
        %pause(0.1)
        drawnow    
    end    


