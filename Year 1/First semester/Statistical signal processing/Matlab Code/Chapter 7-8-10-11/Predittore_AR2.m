%
% Questo programma genera un processo AR(2) ed implementa il Predittore ad
% 1 passo
%
clear all;
close all;
N=1000;
varXdesired=1;
varW=1;
index=[1:N];
w=sqrt(varW)*randn(1,N);
%
% AR(2) process
%
modl=0.1;
freq=1/8;
V(1)=modl*exp(j*2*pi*freq);
V(2)=V(1)';
coeff=poly(V);
%
% Calcolo della funzione di autocorrelazione ACF teorica
%
Am=[1 coeff(2) coeff(3); coeff(2) (1+coeff(3)) 0; -coeff(3) -coeff(2) -1];
Rvect=inv(Am)*[varW 0 0]';
P=[1 1; V(1) V(2)];
co=inv(P)*Rvect(1:2);
mm=[0:64];
Rs=co(1)*V(1).^mm+co(2)*V(2).^mm;
for ii=66:127
    Rs(ii)=Rs(128-ii+1);
end
varX=Rs(1);                % Potenza di X[n] se varW=1
Gamma=varXdesired/varX;
x=filter(1,coeff,w);
xdesired=x.*sqrt(Gamma);   % we force the AR(2) to have the desired power
a=-coeff(2:3)
xhat=filter([0 a],1,xdesired);
%
% Figura rappresentante una possibile realizzazione
%
figure(1)
plot(0:N-1,xdesired,'g','Linewidth', 2); grid on; hold on;
plot(0:N-1,xhat,'r','Linewidth', 2);grid on;
axis([N-100 N -3 3]);
legend('X[n]','1-step prediction of X[n]');
title('X[n] AR(2) process, 1-step prediction of X[n]]')
xlabel('Discrete-Time [n]')
ylabel('X[n], estimate of X[n]')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%