close all 
clear all 

R=0.045; % radius m
V0=2; %V0 Volts 
N=150;
M=100; %NB OBV SE AUMENTO N O M AUMENTO PRECISIONE DELLA GRID QUINDI MEGLIO (RISULTATO PIù SIMILE A QUELLO CON FINITE ELEMENT SU COMSOL) MA TEMPO DI CALCOLO + GRANDE 
d_r=R/N; %radial step 
d_theta=pi/M; %angular step 


theta_0=(5:5:175)*pi/180; %theta_0 to calculate uniform field (same range as in FE example)
indexTheta0=1; 
tic
for k=1:length(theta_0)
    B=sparse((N+1)*(M+1),1);
    A=sparse((N+1)*(M+1),(N+1)*(M+1)); 
    for i=1:N+1 %radial index
        c1=(i-1)^2*d_theta^2;
        c2=(i-1)*d_theta^2;
        k1=c1+c2;
        k2=c1;
        k3=-(2+2*c1+c2);
        for j=1:M+1 %angular index
            h=(M+1)*(i-1)+j;
            if (i==1) %center of the dish
                A(h,h)=M+1;
                A(h,M+1+1:M+1+M+1)=-1;
            elseif (i==N+1) %circular boundary
                theta=(j-1)*d_theta;
                if (theta <= theta_0(k)/2) %active electrode 
                    B(h)=V0;
                    A(h,h)=1;
                elseif (theta >= pi-theta_0(k)/2) %ground electrode
                    A(h,h)=1; %B(index) is 0 by default
                else %electrical insulation
                    A(h,h)=1;
                    A(h,h-(M+1))=-1;
                end
            elseif (j==1) %x-axis (thetaj=0)
                A(h,h)=1;
                A(h,h+1)=-1;
            elseif (j==M+1) %x-axis (thetaj=pi)
                A(h,h)=1;
                A(h,h-1)=-1;
            else   %in the domain
                A(h,h)=k3;  %i,j -> h=(M+1)*(i-1)+j
                A(h,h-(M+1))=k2; %i-1,j  ->h-(M+1)
                A(h,h+(M+1))=k1; %i+1,j -> h+(M+1)
                A(h,(h+1))=1; %i,j+1 -> h+1
                A(h,(h-1))=1; %i,j-1 -> h-1
            end
        end
    end
    %solve the system
    Y=A\B;
    V=reshape(Y,M+1,N+1); %note: r -> cols  ; theta -> rows

    %% Finite difference approximation of Ec
    Ec=abs( (V(1,2)-V(1,1))/d_r );
    %Eratio=E/Ec
    Eratio=ones(size(V)); % r=0 -> center -> E=Ec -> Eratio=1
    E=ones(size(V))*Ec; % r=0 -> center -> E=Ec -> Eratio=1
    %GammaA
    GammaA=zeros(size(V)); 
   
    %aggiornamento ratio, obv per i=1-> r=0 -> centro è di default 1
    for i=2:N+1 %i=1 -> center (i->r->cols j->theta->rows
        r=(i-1)*d_r;
        for j=1:M+1
            theta=(j-1)*d_theta;
            dVdr=(V(j,i)-V(j,i-1))/d_r; %backward approx. of derivative
            if (j==1)||(j==M+1) %theta 0, pi -> zero flux x axis 
                dVdtheta=0;
            else
                dVdtheta=(V(j,i)-V(j-1,i))/d_theta; %backward approx. of derivative (per calcolare modulo E)
            end
            E(j,i)=((sin(theta)*dVdr+cos(theta)*dVdtheta/r)^2+... 
                + (cos(theta)*dVdr-sin(theta)*dVdtheta/r)^2)^0.5;
            Eratio(j,i)=E(j,i)/Ec;

        end
    end
   
     % calculate the area of uniform field
    Acc=0;%accumulator
    for i=1:N+1
        r=(i-1)*d_r; %discretized r
        for j=1:M+1
            if (Eratio(j,i)>=0.9) &&  ( Eratio(j,i)<=1.1 ) 
                GammaA(j,i)=1;
            end
            Acc=Acc+GammaA(j,i)*r*d_r*d_theta; %versione discreta integrale per calcolo area
        end
    end


Area(k)=Acc*1e6*2; %*2 because we worked on half the dish
   

    

end
toc
%% plot & compare with FD

figure;
plot(theta_0,Area); xlabel('theta_0');ylabel('Area (mm^2)'); title('Area of uniform field');
AreaFe=load('areadish.txt'); %results of the COMSOL FE (finite elements) simulation
hold on 
plot(AreaFe(:,1),AreaFe(:,2));
legend('FD','FE')
[MFd iMFd]=max(Area); 
thetaFdMax=theta_0(iMFd)*180/pi %theta_0 max area FD (finite diference)
[MFe iMFe]=max(AreaFe(:,2)); 
thetaFeMax=AreaFe(iMFe,1)*180/pi %theta_0 max area FE 

%vedo che per entrambi theta per cui ho il massimo è 90 gradi


