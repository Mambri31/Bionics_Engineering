%
% Questo programma utilizza stime parametriche per processi AR 
% per stimare lo spettro di un processo ARMA(P,Q)
%
clear all;
close all;
%
% Valore parametri ingresso 
%
N=256;      % Sample size
Ns=10^3;    % Number of Monte Carlo runs
trans=100;
N1=N+trans;
ORDER=16;   % Order of the assumed AR model
%
% Processo a banda larga ARMA(4,4)
% b=[1 -1.3817 1.5632 -0.8843 0.4096];
% a=[1 -0.3544 0.3508 0.1736 0.2401];
%
% Processo a banda stretta ARMA(4,2)
 b=[1 1.5857 0.9604];
 a=[1 -1.6408 2.2044 -1.4808 0.8145];
%
% Calcolo di zeri e poli e grafico polare
%
zero=roots(b);
pole=roots(a);
[THa,Ra] = cart2pol(real(pole),imag(pole));
[THb,Rb] = cart2pol(real(zero),imag(zero));

figure(1)
polar(THa,Ra,'*')
title('Zeros & Poles diagram')
hold on
polar(THb,Rb,'o')
legend('pole','zero')
%
% Calcolo della risposta impulsiva e grafico
%
% Si noti che in entrambi i casi vale h[0]=1,
% Nel caso del processo ARMA(4,2) dobbiamo buttare i primi 2 campioni di
% iir, nel caso di processo ARMA(4,4) no. Non mi è chiaro perché.
% 
% L=0;  % Per ARMA(4,4)
 L=2;  % Per ARMA(4,2)

iir=dimpulse(b,a);
D=size(iir);
h=iir(1+L:D);

figure(2)
stem(0:D-1-L,h,'Linewidth', 2);
axis([0 20 -1.5 1])
grid on;
title('Process: ARMA(4,2) - Impulse response h[n]')
xlabel('Discrete-Time (n)')
ylabel('Impulse response h[n]')
%
% Calcolo della DSP teorica e grafico
%
Nfft=4096;
deltaf=1/Nfft;
f=[0:deltaf:1-deltaf];
x=exp(1i*2*pi*f);
num=polyval(b,x);
den=polyval(a,x);
DSP=(abs(num./den)).^2;

figure(3)
plot(f,10*log10(DSP),'Linewidth', 2);
grid on;
% axis([0 0.5 -10 10])     % per ARMA(4,4) - PSD broadband
axis([0 0.5 -50 40])   % per ARMA(4,2) - PSD narrowband
 
title('Process: ARMA(4,2) - PSD [dB]')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')

figure(4)
plot(f,DSP,'Linewidth',2);
grid on;
title('Process: ARMA(4,2) - PSD')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
% axis([0 0.5 0 10])     % per ARMA(4,4) - PSD broadband
axis([0 0.5 0 2000]) % per ARMA(4,2) - PSD narrowband
%
% Calcolo della funzione di autocorrelazione del processo e grafico
%
Rx=real(ifft(DSP)).';  % Rx[m] è una funzione reale

figure(5)
stem(0:size(Rx)-1,Rx,'Linewidth', 2);grid on;
axis([0 100 -80 120])
legend('Autocorrelation Function (ACF)');
xlabel('Time-lag [m]')
ylabel('ACF')
title('Process: ARMA(4,2) - ACF')
%
% Calcolo del Periodogramma medio e figure
%
for i=1:Ns
    w1=randn(1,N1);
    y1=filter(b,a,w1);
    y1f=y1(trans+1:trans+N);
    period1(i,:)=2*pi*fftshift(pyulear(y1f,ORDER,Nfft,'twosided'));
    perls1(i,:)=2*pi*fftshift(pcov(y1f,ORDER,Nfft,'twosided'));
    param(i,:)=aryule(y1f,ORDER);
end
per1=mean(period1);
perls=mean(perls1);
%
% Posizione dei nuovi poli e figura
%
avparam=mean(param);
poleav=roots(avparam);
[THav,Rav] = cart2pol(real(poleav),imag(poleav));

figure(6)
polar(THav,Rav,'*');
title('Zeros & Poles diagram - Estimated L(z)')
legend('pole')

% Stima della PSD mediata su Ns realizzazioni in scala logaritmica 
figure(7)
plot(f,10*log10(fftshift(per1)),f,10*log10(DSP), 'Linewidth', 2);
grid on;
legend('Estimated PSD - Yule-Walker method','True PSD')
title('Process: ARMA(4,2) - Assumed model: AR(16) - M=10^3 MC runs')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')
axis([0 0.5 -50 40])

% Stima della PSD - Metodo YW - 1 realizzazione - scala logaritmica 
figure(8)
plot(f,10*log10(fftshift(period1(Ns,:))),f,10*log10(DSP), 'Linewidth', 2);
grid on;
legend('Estimated PSD - Yule-Walker method','True PSD')
title('Process: ARMA(4,2) - Assumed model: AR(16)')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')
axis([0 0.5 -50 40])

% Stima della PSD - Periodogramma - Ns realizzazioni - scala lineare
figure(9)
plot(f,fftshift(per1),f,DSP, 'Linewidth', 2);
grid on;
legend('Estimated PSD - Yule-Walker method','True PSD')
title('Process: ARMA(4,2) - Assumed model: AR(16)- M=10^3 MC runs')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
axis([0 0.5 0 2000])

% Stima della PSD con il metodo Least Squares Yule Walker (LS-YW)- 1 realizzazione
figure(10)
plot(f,10*log10(fftshift(perls(1,:))),f,10*log10(DSP),'Linewidth',2);
legend('Least Squares YW','PSD')
title('Process: ARMA(4,2) - Assumed model: AR(16)')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')
grid on
axis([0 0.5 -50 40])

% Stima della PSD, confronto tra metodo YW e LS-YW - 1 realizzazione
figure(11)
plot(f,10*log10(fftshift(perls1(1,:))),f,10*log10(DSP),f,10*log10(fftshift(per1(1,:))),'Linewidth',2);
legend('Least Squares YW','PSD','YW')
title('Process: ARMA(4,2) - Assumed model: AR(16)')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')
grid on
axis([0 0.5 -50 40])

% Stima della PSD mediante Periodogramma - 1 realizzazione
figure(12)
fNzp=[-0.5:1/Nfft:0.5-1/Nfft]';
w1=randn(1,N1);
y1=filter(b,a,w1);
y1f=y1(trans+1:trans+N);
Period=abs(fft(y1f,Nfft)).^2/N;
plot(f,10*log10(Period),f,10*log10(DSP), 'Linewidth', 2);
grid on;
legend('Periodogram','PSD')
title('Narrowband ARMA(4,2) process')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')
grid on
axis([0 0.5 -50 40])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
