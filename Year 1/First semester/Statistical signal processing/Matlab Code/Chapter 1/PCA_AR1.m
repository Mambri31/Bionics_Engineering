%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab genera una realizzazione di un processo AR(1) 
% e traccia i grafici della funzione di autorcorrelazione (ACF) 
% e della densità spettrale di potenza (PSD)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

N=100;               % Numero di campioni generati
rho=0.1 ;            % Coefficiente di correlazione ad un passo
varX=1;              % Potenza del processo AR(1) X[n]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

n=[0:N-1]';
varW=varX*(1-rho^2);                   % Potenza del processo delle innovazioni W[n]
Ntrans=ceil(3*log(10)/abs(log(rho)));  % Si considera la risposta impulsiva nulla quando è minore di 10^-3
Ntot=Ntrans+N;
W=sqrt(varW)*randn(Ntot,1);
Z=filter(1,[1 -rho]',W);
X=Z(Ntrans+1:Ntot);
Rx=varX*rho.^n;
f=[0:0.001:0.5]';
Sx=varW./((abs(1-rho*exp(-j*2*pi*f))).^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(1)
stem(n,X,'Linewidth',2)  
grid on
xlabel('Discrete-Time [n]')               
ylabel('X[n]')
title('Gaussian AR(1) process')
legend('One realization - AR(1) process')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(2)
stem(n,Rx,'Linewidth',2)  
grid on
xlabel('Time-Lag [m]')               
ylabel('R_{X}[m]')
title('Gaussian AR(1) process')
legend('ACF - AR(1) process')
axis([0 20 0 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(3)
plot(f,Sx,'Linewidth',2)  
grid on
xlabel('Digital Frequency (f)')               
ylabel('S_{X}(f)')
title('Gaussian AR(1) process')
legend('PSD - AR(1) process')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T = toeplitz(Rx);
[V,D] = eig(T);
E = dsort(diag(D));

figure(4)
stem(E,'Linewidth',2)  
grid on
xlabel('Order (i)')               
ylabel('Eigenvalues')
title('AR(1)process: Eigenvalues of covariance matrix R_{X} 100x100')
legend('Eigenvalues of R_{X}')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(5)
stem(V(:,100),'Linewidth',2)  % Eigenvector P is the column number N-P+1
grid on
xlabel('Discrete-Time (n)')               
ylabel('Eigenfunctions of random process X[n]')
title('AR(1) process: Eigenvectors of covariance matrix R_{X} 100x100')
legend('Eigenvector 4 of R_{X}')
axis([0 100 -0.2 0.2])



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



