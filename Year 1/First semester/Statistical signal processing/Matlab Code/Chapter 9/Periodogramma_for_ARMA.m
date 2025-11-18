%
% Questo programma calcola il Periodogramma di un processo ARMA(P,Q)
%
clear all;
close all;
Nrun=10^3;   % Numero di stime della PSD generate (Monte Carlo runs)
%
% Processo a banda larga ARMA(4,4)
 b=[1 -1.3817 1.5632 -0.8843 0.4096];
 a=[1 -0.3544 0.3508 0.1736 0.2401];
%
% Processo a banda stretta ARMA(4,2)
% b=[1 1.5857 0.9604];
% a=[1 -1.6408 2.2044 -1.4808 0.8145];
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

 L=0;  % Per ARMA(4,4)
% L=2;  % Per ARMA(4,2)

iir=dimpulse(b,a);
D=size(iir);
h=iir(1+L:D);

figure(2)
stem(0:D-1-L,h,'Linewidth', 2);
axis([0 20 -1.5 1])
grid on;
title('Process: ARMA(4,4) - Impulse response h[n]')
xlabel('Discrete-Time (n)')
ylabel('Impulse response h[n]')
%
% Calcolo della DSP teorica e grafico
%
Nfft=4096;
deltaf=1/4096;
f=[-0.5:deltaf:0.5-deltaf];
x=exp(1i*2*pi*f);
num=polyval(b,x);
den=polyval(a,x);
DSP=(abs(num./den)).^2;

figure(3)
plot(f,10*log10(DSP),'Linewidth', 2);
grid on;
axis([0 0.5 -10 10])
title('Process: ARMA(4,4) - PSD [dB]')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD) [dB]')

figure(4)
plot(f,DSP,'Linewidth', 2);
grid on;
axis([0 0.5 0 10])
title('Process: ARMA(4,4) - PSD')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')

%
% Finestre di Bartlett e grafici
%
Nvect=[64 256 1024];
%
% Generazione dati, calcolo dei periodogrammi medi e figure
%
N1=Nvect(1)+10;
for i=1:Nrun
    w1=randn(1,N1);
    y1=filter(b,a,w1);
    y1f=y1(11:N1);
    period1(i,:)=2*pi*fftshift(periodogram(y1,[],Nfft,'twosided'));
end
per1=mean(period1);
std1=std(period1);
N2=Nvect(2)+10;
for i=1:Nrun
    w2=randn(1,N2);
    y2=filter(b,a,w2);
    y2f=y2(11:N2);
    period2(i,:)=2*pi*fftshift(periodogram(y2,[],Nfft,'twosided'));
end
per2=mean(period2);
std2=std(period2);
N3=Nvect(3)+10;
for i=1:Nrun
    w3=randn(1,N3);
    y3=filter(b,a,w3);
    y3f=y3(11:N3);
    period3(i,:)=2*pi*fftshift(periodogram(y3,[],Nfft,'twosided'));
end
per3=mean(period3);
std3=std(period3);

figure(5)
plot(f,DSP,'magenta-.','Linewidth', 2);grid on;
hold on
plot(f,per1,f,per1+std1,f,per1-std1,'Linewidth', 2);
grid on;
plot(f,DSP,'magenta-.','Linewidth', 2);grid on;
legend('True PSD','Average Periodogram','Average Per.+std','Average Per.-std')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
title('PSD - Process: ARMA(4,4) - Sample size: N=64 - #MC runs=10^3')
grid on
axis([0 0.5 1.2*min(real(per1-std1)) 1.2*max(real(per1+std1))])

figure(6)
plot(f,DSP,'magenta-.','Linewidth', 2);grid on;
hold on
plot(f,per2,f,per2+std2,f,per2-std2,'Linewidth', 2);
grid on;
plot(f,DSP,'magenta-.','Linewidth', 2);grid on;
legend('True PSD','Average Periodogram','Average Per.+std','Average Per.-std')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
title('PSD - Process: ARMA(4,4) - Sample size: N=256 - #MC runs=10^3')
grid on
axis([0 0.5 1.2*min(real(per2-std2)) 1.2*max(real(per2+std2))])

figure(7)
plot(f,DSP,'magenta-.','Linewidth', 2);grid on;
hold on
plot(f,per3,f,per3+std3,f,per3-std3,'Linewidth', 2);
grid on;
plot(f,DSP,'magenta-.','Linewidth', 2);grid on;
legend('True PSD','Average Periodogram','Average Per.+std','Average Per.-std')
xlabel('Digital frequency (f)')
ylabel('Power Spectral Density (PSD)')
title('PSD - Process: ARMA(4,4) - Sample size: N=1024 - #MC runs=10^3')
grid on
axis([0 0.5 1.2*min(real(per3-std3)) 1.2*max(real(per3+std3))])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
