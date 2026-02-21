%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola la stima LMMSE del vettore di segnale S dato il vettore degli osservati X
% X=S*h+W, S[n] è modellato come un processo AR(1), W[n] è AWGN, h[n] è un filtro FIR(Nh-1)
% Stimatore LMMSE 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ns=512;                 % Numero di campioni osservati
Nh=20;                  % Ordine del filtro FIR = Nh-1 (canale LTI)
alfa=0.9;               % Parametro filtro FIR
varS=5;                 % Varianza del segnale
rho=0.8;                % Signal correlation coefficient - AR(1) process
GammadB=10;             % Rapporto varianza segnale - varianza rumore in decibel [dB] 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gamma=10^(GammadB/10);             % Gamma in scala lineare 
varW=varS/Gamma;                   % Potenza del rumore AWGN
h=alfa.^[0:Nh-1]';                 % Risposta impulsiva filtro FIR(Nh-1) che modella il canale LTI
varI=varS*(1-rho^2);               % Potenza del processo delle innovazioni I[n]
Lc=ceil(3*log(10)/abs(log(rho)));  % Si considera la risposta impulsiva nulla quando minore di 10^-3
Ntot=Lc+Ns;                        % numero di campioni dagenerare per averne Ns "corretti"
N=Ns+Nh-1;                         % Numero campioni osservati

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Inn=sqrt(varI)*randn(Ntot,1); % vettore del processo bianco in ingresso al filtro IIR(1)
S0=filter(1,[1 -rho]',Inn);   % uscita dal filtro innovazione IIR(1) di S[n]
S=S0(Lc+1:Ntot);              % vettore dati Nsx1 processo S[n] AR(1) 
X=conv(h,S);                  % segnale X[n] all'uscita del canale LTI 
W=sqrt(varW)*randn(N,1);      % Generazione vettore di rumore AWGN
Z=X+W;                        % Vettore dei dati osservati Z[n]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Cw=varW*eye(N);                       % Noise covariance matrix NxN
Cs=varS*toeplitz((rho.^[0:Ns-1]')');  % Signal covariance matrix NsxNs
ch=zeros(N,1);
rh=zeros(Ns,1);
ch(1:Nh)=h;
rh(1)=1;
H=toeplitz(ch,rh);                    % Channel matrix NxNs
Shat=Cs*H'*(inv(H*Cs*H'+Cw))*Z;       % LMMSE estimate of S given Z

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure (1);
plot([0:Ns-1],S,'r-', 'Linewidth', 2);
grid on;
hold on;
plot([0:N-1],Z,'b', 'Linewidth', 2);
legend('Signal of interest S[n]','Observed signal Z[n]');
title('Z[n] and S[n]');xlabel('Discrete-Time [n]');ylabel('Z[n] and S[n]')
axis([0 N-1 -30 30])

figure (2);
plot([0:Ns-1],S,'r-', 'Linewidth', 2);
grid on;
hold on;
plot([0:Ns-1],Shat,'g', 'Linewidth', 2);
legend('Signal of interest S[n]','LMMSE estimate of S[n]');
title('S[n] and its LMMSE estimate');xlabel('Discrete-Time [n]');ylabel('S[n]')
axis([0 Ns-1 -8 8])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Autore: 
% Fulvio Gini
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: fulvio.gini@unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 