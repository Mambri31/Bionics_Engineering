%
% Questo programma genera un processo AR(2), calcola e disegna la Densità Spettrale di Potenza (PSD) e
% la funzione di autocorrelazione (ACF)
%
clear all;
close all;
N=100;
varW=1;
index=[1:N];
w=sqrt(varW)*randn(1,N);
%
% AR(2) process
%
modl=0.99;
freq=1/4;
V(1)=modl*exp(j*2*pi*freq);
V(2)=V(1)';
coeff=poly(V);
x=filter(1,coeff,w);
%
% Figura rappresentante una possibile realizzazione
%
figure(1)
stem(0:N-1,x,'Linewidth', 2);grid on;
legend('Realizzazione di X[n]');
title('Realizzazione di X[n], processo AR(2)')
xlabel('Discrete-Time [n]')
ylabel('X[n]')
grid on
%
% Calcolo della PSD
%
Nf=1024;
deltaf=1/Nf;
f=[-0.5:deltaf:0.5-deltaf];
xpol=exp(-j*2*pi*f);
den=abs(polyval(coeff,xpol)).^2;
Sxx=varW./den;
%
% Figura della PSD
%
figure(2)
plot(f,Sxx,'Linewidth', 2);grid on;
axis([0 0.5 0 1.01*max(real(Sxx))])
legend('PSD di X[n]');
title('Processo AR(2) - PSD')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
grid on
%
% Calcolo della funzione di autocorrelazione ACF tramite sistema
%
Am=[1 coeff(2) coeff(3); coeff(2) (1+coeff(3)) 0; -coeff(3) -coeff(2) -1];
Rvect=inv(Am)*[varW 0 0]';
P=[1 1; V(1) V(2)];
co=inv(P)*Rvect(1:2);
mm=[0:64];
Rs=co(1)*V(1).^mm+co(2)*V(2).^mm;
for ii=66:127
    Rs(ii)=Rs(128-ii+1);
end
%
% Calcolo della funzione di autocorrelazione tramite IFFT della PSD
%
Rx=fftshift(real(ifft(fftshift(Sxx))));
%
% figura della ACF
%
figure(3)
subplot(2,1,1);
stem(-Nf/2:Nf/2-1,real(Rx),'Linewidth', 2);grid on;
axis([-0.5 50 1.2*min(real(Rx)) 1.2*max(real(Rx))])
ylabel('ACF')
legend('ACF di X[n]');
title('Processo AR(2)- IFFT della PSD')
grid on
subplot(2,1,2);
stem(-63:63,fftshift(Rs),'Linewidth', 2);grid on;
axis([-0.5 50 1.2*min(real(Rx)) 1.2*max(real(Rx))])
legend('ACF di X[n]');
xlabel('Time-lag [m]')
ylabel('Funzione di Autocorrelazione (ACF)')
title('Processo AR(2)- ACF analitica')
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%