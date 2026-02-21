%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab grafica lo spettro di 3 oscillazioni cosinusoidali in rumore AWGN 
% e deriva la stima della due frequenze usando il metodo di Yule Walker basato su modelli AR
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

B=20;                        % Banda filtro passa-basso anti-aliasing in KHz
T= 10;                       % Intervallo di osservazione misurato in millisecondi (msec)
Tc=1/(2*B);                  % Intervallo di campionamento misurato in millisecondi (msec)
N=round(2*B*T);              % Numero di campioni N=T/Tc=2BT deve essere >>1
Nzp=max(N,2^19);             % zero-padding per FFT
Av=[10 5 10]';               % Vettore delle ampiezze
Thetav=2*pi*[0 1/3 1/5]';    % Vettore delle fasi in radianti
Fv=[5 5+2/T 6]';           % Vettore delle frequenze in KHz
FvDig=Fv*Tc;                 % Vettore frequenzi digitali (adim) 
SNR1dBin=10;                 % Rapporto segnale_1-rumore in decibel [dB]
SNR1=10^(SNR1dBin/10);       % SNRin in scala lineare
N0=(Av(1)^2)/(2*B*SNR1);     % Densità spettrale di Potenza monolatera del processo di rumore bianco  
varW=N0/2;                   % Potenza di rumore

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=[0:N-1]';                              % Vettore del tempo discreto
s=zeros(N,1);
for i=1:3
s=s+Av(i)*sqrt(1/(2*B))*cos(2*pi*FvDig(i)*k+Thetav(i));   % Generazione del vettore di segnale
end
W=sqrt(varW)*randn(N,1);                 % Generazione del vettore Nx1 di rumore
Y=s+W;                                   % Generazione del vettore Nx1 dei dati osservati 
fx=2*B*(0:Nzp/2)'/Nzp;                   % Vettore delle frequenze analogiche (KHz)   
Yf=fft(Y,Nzp);                           % FFT di Y   
Yf=Yf(1:Nzp/2+1);                        % FFT di Y nel primo semi-periodo
PeriodogramY=abs(Yf).^2./N;              % Periodogramma di Y 
PerdB=10*log10(PeriodogramY/max(PeriodogramY));   % Periodogramma di Y in dB normalizzato

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ORDER=200;
Deltaf=1/Nzp;
f=2*B*[0:Deltaf:1-Deltaf];
YWPSD=(2*pi*fftshift(pyulear(Y,ORDER,Nzp,'twosided')));
YWPSD=YWPSD/max(YWPSD);
YWPSDdB=10*log10(YWPSD);
param=aryule(Y,ORDER);
poleav=roots(param);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure (1);
plot(fx,PerdB,'r','LineWidth',2);
grid on; hold on;
plot(f,fftshift(YWPSDdB),'LineWidth',2);
legend('Periodogram','Yule Walker AR');
title('YW-AR vs Periodogram');xlabel('Analog Frequency f [KHz]');ylabel('PSD [dB]')
axis([0 20 -50 1])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 