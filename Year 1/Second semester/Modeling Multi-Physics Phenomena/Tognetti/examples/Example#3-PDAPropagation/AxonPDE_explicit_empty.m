%Propagation of the action potential (PDA) in an unmyelinated axon

clear variables 
close all
clc

anim=1; %1 to run the animation


%% problem data

L = 5; % axon length (cm)
t_sim=150; % simulation time in msec


Cm = 1; % uF/cmˆ2
GNa_ = 120; % mS/cmˆ2 
GK_ = 36; % mS/cmˆ2
GL = 0.300; % mS/cmˆ2
sigma_i = 1.6667; % mS/cm    
a = 0.0025; % cm
Vr=-60; %membrane rest potential (mV)
VNa=55; %absolute value of Na reversal potential  (mV)
VK=72;  %absolute value of K reversal potential  (mV)
VL=49.387; %Leakage potential  (mV)

x_stim=0.5%0.05 % current stimulation site on the axon (cm)

t_on=2; %current stimulus on (msec)
t_high=1; %duration of the stimulus (msec)
t_off=t_on+t_high; %current stimulus off (msec)
Is=0.1; %current stimulus intensity (uA)

dx = 0.01; % x_increment (cm)
H = round(L/dx); % total spatial nodal points
js=round(x_stim/dx);% spatial nodal index of stimulation
x=(0:dx:L);

dt= 0.01; %t_increments msec (try multyply by 2.5 to obtain a non stable condition
N=round(t_sim/dt); %total temporal nodal points
t=(0:dt:t_sim);
i_on=round(t_on/dt); %temporal nodal index of t_on
i_off=round(t_off/dt); %temporal nodal index of t_off


%%% necessary condition for stability of this method
%Di=(a*sigma_i)/(2*Cm); %diffusivity
%cond=Di*dt/(dx^2) %should be less than 0.5


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

xp=3;
d=0.1;
sigma_o=1;% mS/cm    


for i=1:N %time
    for j=2:H %x -> j=1 and j=H+1 -> zero flux boundary conditions
    
        %Explicit method (iterative): (starting from the initial
        %conditions), at time i we calculate the unknwons at time i*deltat+1
        

        %% calculus of Vm at next time i+1

      
             
    end
       %zero flux boundary
       
end

%extracellular potential at the electrode 

    
figure 
plot(t(1:N),phi_p);

%% plot results

%plot membrane potential on the axon for t=t1
t1=30; it1=round(t1/dt)+1;
figure
plot(x,Vm(it1,:),'LineWidth',2);
title(['t=' num2str(t1) ' msec']);
hold on 
plot(x,Vr*ones(H+1,1),'-.','LineWidth',1.5);
xlabel('[cm]');
ylabel('[mV]');
legend('Vm','Vr');

%plot membrane potential at: (L/4) L/2 3/4L  
x1=L/4;  jx1=round(x1/dx)+1;
x2=L/2;  jx2=round(x2/dx)+1;
x3=3*L/4; jx3=round(x3/dx)+1;

figure
plot(t,Vm(:,jx1),'LineWidth',2);
hold on 
plot(t,Vm(:,jx2),'LineWidth',2);
plot(t,Vm(:,jx3),'LineWidth',2);
%plot(t,Vr*ones(1,N+1),'-.','LineWidth',1.5);
xlabel('[msec]');
ylabel('[mV]');
legend(['x=' num2str(x1) ' cm'],['x=' num2str(x2) ' cm'],['x=' num2str(x3) ' cm']);

%% animate results
if anim==1
    nframe=200;
    DT=round((N-1)/nframe); % to plot every nframe frames
    figure
    a1=animatedline('Color','b','LineWidth',2);
    axis([x(1) x(end) -90 50]); 
    ylabel('[mV]');
    xlabel('[cm]');
    legend('Vm');
    for i=1:DT:N
        clearpoints(a1);
        addpoints(a1,x,Vm(i,:));
        drawnow    
    end    
end

