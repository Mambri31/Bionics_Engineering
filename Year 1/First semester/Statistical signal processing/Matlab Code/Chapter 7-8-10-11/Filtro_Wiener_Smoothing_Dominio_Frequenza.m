%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola la risposta in frequenza del filtro di
% Wiener di tipo smoothing e la confronta con la PSD del processo di
% segnale S[n], assumendo rumore additivo bianco.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

varS=1;                       % Varianza del segnale
rho=-0.9;                     % Signal correlation coefficient - AR(1) process
GammadB=[-10 -5 0 5 10]';     % Rapporto varianza segnale - varianza rumore in decibel [dB] 
Gamma=10.^(GammadB/10);
Ngamma=length(GammadB);
f=[0:0.01:0.5]';
Nf=length(f);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H=(1-rho^2).*Gamma(1)./((1-rho^2).*Gamma(1)+1+rho^2-2*rho*cos(2*pi*f));

Ss=varS*(1-rho^2)./(1+rho^2-2*rho*cos(2*pi*f));

for k=2:Ngamma

H=[H (1-rho^2)*Gamma(k)./((1-rho^2)*Gamma(k)+1+rho^2-2*rho*cos(2*pi*f))];

end

figure (1);
plot(f,Ss/max(Ss),'r-.', 'Linewidth', 2);
grid on;
hold on;
plot(f,H, 'Linewidth', 2);
legend('PSD di S[n]','H(f), \gamma=-10dB', 'H(f), \gamma=-5dB', 'H(f), \gamma=-0dB', 'H(f), \gamma=5dB', 'H(f), \gamma=10dB');
title('PSD di S[n] e H(f) filtro di Wiener per \rho = -0.9');
xlabel('Digital Frequency (f)');ylabel('PSD di S[n] e H(f)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 