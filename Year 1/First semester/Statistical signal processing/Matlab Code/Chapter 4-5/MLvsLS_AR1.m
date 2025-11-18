%
% Questo programma calcola l'MSE dello stimatore ML e dello stimatore LS in
% rumore additivo Gaussiano correlato con matrice di covarianza R
%
clear all;
close all;

% Derivazione del MSE in funzione del parametro rho

N=100;        % Numero di campioni generati
F0=0.01;       % Frequenza digitale della cosinusoide
varW=1;        % Potenza di W[n]
n=[0:N-1]';
h1=cos(2*pi*F0*n);
h2=sin(2*pi*F0*n);
H=[h1 h2];
rhov=[-0.99:0.01:0.99]'; % One-lag correlation coefficient di W[n] processo AR(1)
Nv=length(rhov);
MSE_ML_Theta2=zeros(Nv,1);
MSE_LS_Theta2=zeros(Nv,1);

for k=1:Nv
    
rho=rhov(k);
r=rho.^n;  % Prima colonna della matrice di Toeplitz R
R=varW*toeplitz(r);     % R = matrice di covarianza di W[n]
Rinv=inv(R);        % inversa della matrice di covarianza R
MSE_ML=inv((H'*Rinv*H));  % Matrice dell'MSE dello stimatore ML
MSE_LS=inv(H'*H)*(H'*R*H)*inv(H'*H);  % Matrice dell'MSE dello stimatore LS
MSE_ML_Theta2(k)=10*log10(MSE_ML(2,2));
MSE_LS_Theta2(k)=10*log10(MSE_LS(2,2));

end

figure(1)
grid on;
plot(rhov,MSE_ML_Theta2,'b','Linewidth', 2);
hold on;
plot(rhov,MSE_LS_Theta2,'g-','Linewidth', 2);
legend('MSE ML estimator of \Theta_2', 'MSE LS estimator of \Theta_2');
xlabel('One-lag correlation coefficient \rho')
ylabel('MSE [dB]')
title('MSE of the ML and LS estimators in correlated noise')

% Derivazione del MSE in funzione del numero dei dati N

Nv=[2:1:100]';        % Numero di campioni generati
F0=0.3;       % Frequenza digitale della cosinusoide
varW=1;        % Potenza di W[n]
rho=0.99; % One-lag correlation coefficient di W[n] processo AR(1)
NL=length(Nv);
MSE_ML_Theta2=zeros(NL,1);
MSE_LS_Theta2=zeros(NL,1);

for k=1:NL
    
N=Nv(k);
n=[0:N-1]';
h1=cos(2*pi*F0*n);
h2=sin(2*pi*F0*n);
H=[h1 h2];
r=rho.^n;  % Prima colonna della matrice di Toeplitz R
R=varW*toeplitz(r);     % R = matrice di covarianza di W[n]
Rinv=inv(R);        % inversa della matrice di covarianza R
MSE_ML=inv((H'*Rinv*H));  % Matrice dell'MSE dello stimatore ML
MSE_LS=inv(H'*H)*(H'*R*H)*inv(H'*H);  % Matrice dell'MSE dello stimatore LS
MSE_ML_Theta2(k)=10*log10(MSE_ML(2,2));
MSE_LS_Theta2(k)=10*log10(MSE_LS(2,2));

end


figure(2)
grid on;
plot(Nv,MSE_ML_Theta2,'b','Linewidth', 2);
hold on;
plot(Nv,MSE_LS_Theta2,'g-','Linewidth', 2);
legend('MSE ML estimator of \Theta_2', 'MSE LS estimator of \Theta_2');
xlabel('Data size (N)')
ylabel('MSE [dB]')
title('MSE of the ML and LS estimators in correlated noise')

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%