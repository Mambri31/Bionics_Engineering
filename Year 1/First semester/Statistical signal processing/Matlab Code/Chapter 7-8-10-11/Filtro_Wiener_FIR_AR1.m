%
% Questo programma calcola e disegna la funzione di autocorrelazione di un
% processo AR(1) e implementa il filtro di Wiener FIR 
%
clear all;
close all;

N=1000;       % Numero di campioni generati
LACF=200      % lag massimo a cui si calcola la ACF
P=2;         % ordine del filtro FIR di Wiener
varS=1;       % Potenza di S[n]
gammadb=-3;    % SNR in dB
rho=-0.5;     % One-lag correlation coefficient di S[n] processo AR(1)

varI=varS*(1-rho^2);     % Potenza di I[n] processo delle innovazioni di S[n]
gamma=10^(gammadb/10);   % SNR in scala lineare
varW=varS/gamma;         % Potenza di W[n]
varX=varS+varW;          % Potenza di X[n]

DiscreteTime=[0:N-1];
Inn=sqrt(varI)*randn(N,1);
W=sqrt(varW)*randn(N,1);
%
% S[n] = AR(1) Gaussiano, W[n] = AWGN
%
a(1)=1;
a(2)=-rho;
b(1)=1;
S=filter(b,a,Inn);
X=S+W;
timelag=[0:LACF]';
Rs=varS*rho.^timelag;  
%
% Figura della ACF
%
figure(1)
stem(timelag,Rs, 'Linewidth', 2);grid on;
axis([-0.1 LACF min(Rs) 1.1*max(Rs)]); 
legend('ACF di S[n]');
xlabel('Time-lag [m]')
ylabel('Funzione di Autocorrelazione (ACF)')
title('S[n] = processo Gaussiano AR(1)')
grid on;
%
% Implementazione del filtro di Wiener FIR(P)
%
m=[0:P]';
cs=varS*rho.^m;                % vettore della ACF di S[n]
CW=varW*eye(P+1);             % matrice di covarianza di W[n]
CS=varS*toeplitz((rho.^m)');  % matrice di covarianza di S[n]
CX=CS+CW;                     % matrice di covarianza di X[n]
CXinv=inv(CX);                % inversa della matrice di covarianza di X[n]
h=CXinv*cs;                   % Risposta impulsiva del filtro di Wiener FIR(P)
Shat=filter(h,1,X);           % Uscita del filtro di Wiener FIR(P)
%
MSE=varS-h'*cs;
%
figure(2)
mm=[0:2*P]';
hh=zeros(2*P+1,1);
hh(1:P+1)=h;
stem(mm,hh, 'Linewidth', 2);grid on;
axis([-0.1 2*P  min(hh) 1.1*max(hh)]); 
legend('Risposta impulsiva del filtro di Wiener FIR, h[n]');
xlabel('Discrete-Time [n]')
ylabel('h[n]')
title('Risposta impulsiva del filtro di Wiener FIR, h[n]')
grid on;

figure(5)
Nz=1024;
H=fft(h,Nz);
f=[0:1/Nz:0.5]';
Nf=length(f);
AH=abs(H(1:Nf));
plot(f,AH,'b','Linewidth', 2);grid on;
legend('Risposta in Frequenza del filtro di Wiener FIR, H(f)');
xlabel('Digital Frequency (f)')
ylabel('|H(f)|')
title('Risposta in Frequenza del filtro di Wiener FIR')
grid on;

% Realizzazione di S[n] e X[n]
%
figure(3)
plot(DiscreteTime,S,'r','Linewidth', 2);grid on;
hold on
plot(DiscreteTime,X,'b','Linewidth', 2);grid on;
legend('S[n]', 'X[n]');
axis([N-100 N -3*sqrt(varX) 3*sqrt(varX)]); 
xlabel('Discrete-Time [n]')
ylabel('S[n], X[n]')
title('X[n]=S[n]+W[n], S[n] = processo Gaussiano AR(1), W[n] = AWGN')
%
% Realizzazione di S[n] e della stima di S[n]
%
figure(4)
plot(DiscreteTime,S,'r','Linewidth', 2);grid on;
hold on
plot(DiscreteTime,Shat,'b','Linewidth', 2);grid on;
legend('S[n]', 'Stima di S[n]');
axis([N-100 N -3*sqrt(varS) 3*sqrt(varS)]); 
xlabel('Discrete-Time [n]')
ylabel('S[n], Stima di S[n]')
title('X[n]=S[n]+W[n], S[n] = processo Gaussiano AR(1), W[n] = AWGN')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%