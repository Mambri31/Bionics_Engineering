%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola la stima LMMSE di S in forma ricorsiva 
% e ne traccia il grafico in funzione di N
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varS=1;                   % Varianza del segnale
etaS=1;                   % Valor medio del segnale

% Per la figura Stima_S ed MSE vs. N

Nmax=100000;                 % Max numero di campioni osservati
DiscreteTime=[0:Nmax-1]';   % Vettore del tempo-discreto
GammadB=10;                % Rapporto varianza segnale - varianza rumore in decibel [dB] 
Gamma=10^(GammadB/10);      % Gamma in scala lineare 
varW=varS/Gamma;            % Potenza di rumore 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S=etaS+sqrt(varS)*randn(1,1);

T0=Nmax/2;   % Periodo della variazione cosinusoidale intorno al valore costante S
eps=0.3;     % Ampiezza dell oscostamento, per generare il segnale S costante basta mettere eps=0

Stv=S+eps*cos(2*pi*DiscreteTime/T0); % Si ipotizza che il segnale cambi nel tempo (lentamente) in modo cosinusoidale 

W=sqrt(varW)*randn(Nmax,1);
X=Stv+W;

Shat0=etaS;
MSE0=varS;

K(1)=MSE0/(MSE0+varW);
Shat(1)=Shat0+K(1)*(X(1)-Shat0);
MSE(1)=(1-K(1))*MSE0;

for n=2:Nmax

    K(n)=MSE(n-1)/(MSE(n-1)+varW);
    Shat(n)=Shat(n-1)+K(n)*(X(n)-Shat(n-1));
    MSE(n)=(1-K(n))*MSE(n-1);

end

MSEdB=10*log10(MSE);

figure (1);
semilogx(DiscreteTime,MSEdB,'b', 'Linewidth', 2);
grid on;
legend('MSE stimatore LMMSE di S');
title('MSE vs N');xlabel('Sample size (N)');ylabel('MSE [dB]')

figure (2);
% semilogx(DiscreteTime,Shat,'b', 'Linewidth', 2);
plot(DiscreteTime,Shat,'b', 'Linewidth', 2);
grid on;
hold on;
% semilogx(DiscreteTime,Stv,'r-.','Linewidth', 2);
plot(DiscreteTime,Stv,'r-.','Linewidth', 2);
legend('Stima LMMSE di S','Valore vero di S');
title('Stima vs N');xlabel('Sample size (N)');ylabel('Stima LMMSE di S')

figure (3);
plot(DiscreteTime(1:100),X(1:100),'b', 'Linewidth', 2);
grid on;
hold on;
plot(DiscreteTime(1:100),Stv(1:100),'r-.','Linewidth', 2);
legend('Segnale osservato X[n]','Segnale costante S');
title('Segnali X[n] ed S vs N');xlabel('Sample size (N)');ylabel('Segnali X[n] ed S')
axis([0 99 -10 10])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 