%
% Questo programma implementa il filtro di Wiener IIR Causale 
% per un processo AR(1)
%
clear all;
close all;

N=1000;        % Numero di campioni generati
Nh=20;         % Lunghezza del grafico della risposta impulsiva
varS=2;        % Potenza di S[n]
gammadb=0;     % SNR in dB
rho=0.8;       % One-lag correlation coefficient di S[n] processo AR(1)

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
S=filter(1,[1 -rho],Inn);
X=S+W;
%
% Implementazione del filtro di Wiener IIR Causale
%
k=1+rho^2+gamma*(1-rho^2);
alpha=(k-sqrt(k^2-4*rho^2))/(2*rho);
c0=gamma*(1-rho^2)*((1-alpha)^2)/((1-alpha*rho)*(gamma*(1-rho^2)+(1-rho)^2));
Shat=filter(c0,[1 -alpha],X);   % Uscita del filtro di Wiener IIR causale
%
MSE=varS*(1-c0/(1-alpha*rho));
%
% Risposta impulsiva del filtro di Wiener IIR Causale
%
figure(1)
n=[0:Nh]';
h=c0*(alpha.^n);
stem(n,h,'Linewidth', 2);grid on;
axis([-0.1 Nh  min(h) 1.1*max(h)]); 
legend('Risposta impulsiva del filtro di Wiener IIR causale, h[n]');
xlabel('Discrete-Time [n]')
ylabel('h[n]')
title('Risposta impulsiva del filtro di Wiener IIR causale, h[n]')
grid on;

% Realizzazione di S[n] e X[n]
%
figure(2)
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
%

f=[0:0.01:0.5]';
Ss=varS*(1-rho^2)./(1+rho^2-2*rho*cos(2*pi*f));
Sw=varW*ones(length(f),1);
Sx=Ss+Sw;
H=abs(c0./(1-alpha*exp(-j*2*pi*f)));

figure (5);
plot(f,10*log10(Ss),'r', 'Linewidth', 2);
grid on;
hold on;
plot(f,10*log10(Sx),'g', 'Linewidth', 2);
plot(f,20*log10(H), 'b','Linewidth', 2);
legend('PSD di S[n]', 'PSD di X[n]','|H(f)|');
title('PSD di S[n], PSD di X[n] e risposta in ampiezza |H(f)| del filtro di Wiener IIR causale');
xlabel('Digital Frequency (f)');
ylabel('PSD di S[n], PSD di X[n] e |H(f)| in dB')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%