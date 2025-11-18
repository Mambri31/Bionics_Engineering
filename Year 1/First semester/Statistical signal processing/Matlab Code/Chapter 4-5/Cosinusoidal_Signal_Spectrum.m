%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calculates and plots the spectrum of  cosinusoidal
% pulse of duration T.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

Theta=0.1;           % Fase iniziale in radianti
f0=40;               % Frequenza analogica dell'oscillazione (misurata in KHz)
fa=[39:0.001:41]';   % Vettore delle frequenze analogiche
T=20;                % Intervallo di osservazione (misurato in msec)


Sfpos=T/2*sinc((fa-f0)*T)*exp(j*Theta).*exp(-j*pi*(fa-f0)*T);
Sfneg=T/2*sinc((fa+f0)*T)*exp(-j*Theta).*exp(-j*pi*(fa+f0)*T);
Sf=abs(Sfpos+Sfneg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
figure(1)
plot(fa,10*log10(Sf),'Linewidth',2)
grid on
hold on;
xlabel('Analog Frequency (f) [KHz]')               
ylabel('Spectrum |P(f)| [dB]')
title('Cosinusoidal Signal Spectrum')
legend('|P(f)|')
axis([39 41 -40 20])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



