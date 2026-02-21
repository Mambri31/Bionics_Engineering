%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab genera M realizzazioni di due v.a. IID uniformemente distribuite in [0,1] 
% e ricava la stima dei parametri della somma V e della differenza Z
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

M=10^5;

U=rand(M,2);         % genera M coppie di v.a. IID uniformemente distirbuite in [0,1]
etaU=0.5;
X=U(:,1);
Y=U(:,2);
Z=X-Y;
V=X+Y;

step=10;  % Passo a cui si calcola e si grafica la stima dei parametri

for k=1:M/step,  % stime calcolate per numero di campioni variabile a passi di "step"
    ncamp=k*step;
    etaZhat(k)=mean(Z(1:ncamp));  
    etaVhat(k)=mean(V(1:ncamp));
    varZhat(k)=var(Z(1:ncamp));  
    varVhat(k)=var(V(1:ncamp));
    covZV=mean((Z(1:ncamp)-etaZhat(k)).*(V(1:ncamp)-etaVhat(k)));
    rhoZVhat(k)=covZV/sqrt( varZhat(k)*varVhat(k));
end

Ncamp=step*[1:M/step]';                          
figure(1)
semilogx(Ncamp,etaZhat,'r',Ncamp,etaVhat,'g',Ncamp,varZhat,'m',Ncamp,varVhat,'k',Ncamp,rhoZVhat,'b', 'Linewidth',2)
grid on;
xlabel('Sample Size (N)');
ylabel('Sample Estimates of moments of Z and V')
title('Sample Estimates of moments of Z=X-Y and V=X+Y, X and Y IID Uniform in [0,1]')
legend('Sample Mean Z','Sample Mean V','Sample Variance Z','Sample Variance V','Sample Correlation Coefficient')
axis([step M -1 3]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

