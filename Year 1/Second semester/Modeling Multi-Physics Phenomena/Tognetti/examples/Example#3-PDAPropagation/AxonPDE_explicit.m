%Propagation of the action potential (PDA) in an unmyelinated axon

clear variables 
close all
clc

anim=0; %1 to run the animation


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

x_stim=1%0.05 % current stimulation site on the axon (cm)

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

        GNa=GNa_*(m(i,j)^3)*h(i,j);
        GK=GK_*(n(i,j)^4);
        INa=GNa*(Vm(i,j)-VNa); %Na
        IK=GK*(Vm(i,j)+VK); %K
        IL=GL*(Vm(i,j)+VL); %L
        Iion=IK+INa+IL; % cIion 
        

        d2Vm=(Vm(i,j+1)-2*Vm(i,j)+Vm(i,j-1))/dx^2;

        if (j==js) %current stimulation point
            if (i>i_on)&&(i<i_off)
                Ie=Is;
            else
                Ie=0;
            end
            Vm(i+1,j)=Vm(i,j)+dt/Cm *(sigma_i*a*d2Vm/2-Iion+Ie/(2*pi*a*dx));
            ImA(i,j)=(2*pi*a*dx)*(a*sigma_i/2)*d2Vm+Ie;
            
        else
            Vm(i+1,j)=Vm(i,j)+dt/Cm *(sigma_i*a*d2Vm/2-Iion);
            ImA(i,j)=(2*pi*a*dx)*(a*sigma_i/2)*d2Vm;
        end
      
     

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

%extracellular potential at the electrode 
acc=0;
for i=1:N
    acc=0;
    for j=1:H
        r=(d^2+(x(j)-xp)^2)^0.5;
        acc=acc+(1/(4*pi*sigma_o))*(ImA(i,j)/r)*dx;
    end
    phi_p(i)=acc;
end
    


%% plot results

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
        pause(0.1)
        drawnow    
    end    
end


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


%% extracellular potential
figure 
plot(t(1:N),phi_p);