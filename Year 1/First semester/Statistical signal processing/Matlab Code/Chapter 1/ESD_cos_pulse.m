%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab la densità spettrale di energia (ESD) del segnale utile: segnale cosinusoidale 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

N=64;
Nzp=2048;
F0=0.2;
A=1;
Theta=pi/3;
varW=2;
n=[0:N-1]';
F=[0:1/Nzp:0.5]';

s=A*cos(2*pi*F0*n+Theta);
Appo1=fft(s,Nzp);
Appo2=(abs(Appo1).^2);
S=Appo2(1:Nzp/2+1)/N;
W=varW*ones(Nzp/2+1,1);

figure(1)
plot(F,S,'b','Linewidth',2)
grid on
hold on;
plot(F, W,'r','Linewidth',2)
xlabel('Digital Frequency (F)')               
ylabel('Power Spectral Density (PSD)')
title('Cosinuosidal signal + AWGN')
legend('PSD Cosinusoidal Signal', 'PSD White Noise')


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



