%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab genera una realizzazione del segnale osservato: segnale Gaussiano costante in rumore AWGN
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

N=10;         % Numero di campioni prelevati
upsampl=50;   % fattore di sovracampionamento per simulare il "tempo continuo"

etaS=10;       % valore medio di S
varS=4;        % varianza di S
gammadB=3;     % valore di Gamma in dB
gamma=10^(gammadB/10);

varW=varS/gamma;  % varianza di W


S=sqrt(varS)*randn(1,1)+etaS       % S con ddp Gaussiana
Wct=sqrt(varW)*randn(N*upsampl,1); % Rumore "tempo continuo" AWGN
Yct=S+Wct;                         % Segnale tempo-continuo osservato
Y=Yct(1:upsampl:N*upsampl);        % Campioni del segnale osservato

figure(1)
plot(Yct,'r','Linewidth',1)      
hold on;
stem([1:upsampl:N*upsampl],Y,'Linewidth',2)
plot([0 N*upsampl],[S S],'g','Linewidth',2)
xlabel('Time (t)')               
ylabel('Y_{B}(t)')
title('Constant Gaussian signal in AWGN')
legend('Observed signal Y_{B}(t)','Sampled signal Y_{B}(nT_{c})','Constant Gaussian signal S')
axis([0 N*upsampl etaS-5*sqrt(varS) etaS+5*sqrt(varS)])

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it



