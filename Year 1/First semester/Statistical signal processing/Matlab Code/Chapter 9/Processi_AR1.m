%
% Questo programma calcola e disegna la funzione di autocorrelazione e la
% PSD di un processo AR(1)
%
clear all;
close all;
N=100;
sigmaw=1;
index=[1:N];
w=sigmaw*randn(1,N);
%
% AR(1) process
%
ro=0.5;         % One-lag correlation coefficient
a(1)=1;
a(2)=-ro;
b(1)=1;
x=filter(b,a,w);
sigmax=sigmaw/sqrt(1-ro^2);    % standard deviation of the AR(1) process
indcor=[0:64];
Rx=sigmax^2*ro.^indcor;         % Autocorrelation function
Nf=1024;
deltaf=1/Nf;
f=[-0.5:deltaf:0.5-deltaf];
Sx=sigmaw^2./(1+ro^2-2*ro.*cos(2*pi*f));   %PSD
V=[ro];
coeff=poly(V)
xpol=exp(-j*2*pi*f);
den=abs(polyval(coeff,xpol)).^2;
Sxx=sigmaw^2./den;
%
% Figure of the process realizations
%
figure(1)
% plot(index,w,index,x,'Linewidth', 2);grid on;
% legend('Innovation Process W[n]','AR(1) Process X[n]');
stem(index,x,'Linewidth', 2);grid on;
legend('Gaussian AR(1) process, \rho =0.5');
axis([0 N -3*sigmax 3*sigmax]); 
xlabel('Discrete-Time [n]')
ylabel('X[n]')
title('Gaussian AR(1) process, \rho =0.5')

% Figure of the correlation
%
figure(2)
stem(indcor,Rx, 'Linewidth', 2);grid on;
axis([0 length(indcor) -sigmax^2 sigmax^2]); 
legend('ACF di X[n]');
xlabel('Time-lag [m]')
ylabel('Autocorrelation Function (ACF)')
title('Gaussian AR(1) process, \rho =0.5')
grid on;
axis([0 20 -1 2 ])
%
% Figure of PSD
%
figure(3)
plot(f,Sxx, 'Linewidth', 2);grid on;
legend('PSD di X[n]');
axis([0 0.5 0 max(Sx)]); 
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
title('Gaussian AR(1) process, \rho =0.5')
grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
