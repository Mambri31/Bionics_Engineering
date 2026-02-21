%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola il Mean Square Error (MSE) degli stimatori
% ML di ampiezza, fase e frequenza di un oscillazione cosinusoidale immersa in rumore
% AWGN e ne fa il grafico al variare di SNRdB. Si riporta anche l'MSE
% delllo stimatore della frequenza mediante il metodo dei momenti (MM)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M=5*10^2;                    % Numero di prove Monte Carlo per ogni SNR
A=10;                        % Ampiezza del segnale
Theta=2*pi/3;                % Fase iniziale in radianti
f0=2;                        % frequenza analogica in hertz (KHz)
B=20;                        % Banda filtro passa-basso anti-aliasing in hertz (KHz)
T= 100;                      % intervallo di osservazione misurato in secondi (msec)
Tc=1/(2*B);                  % intervallo di campionamento misurato in secondi (msec)
N=round(2*B*T);              % Numero di campioni N=T/Tc=2BT deve essere >>1
Nzp=max(N,2^15);             % zero-padding per FFT
F0=f0*Tc;                    % frequenza digitale (adim)                         
SNRdB=[-10:5:40];            % Vettore dei rapporti segnale-rumore in decibel [dB]
SNR=10.^(SNRdB./10);         % Vettore di SNR in scala lineare
N0=(A^2)./(2*B*SNR);         % Densità spettrale di Potenza monolatera del processo di rumore bianco  
varW=N0/2;                   % Vettore delle potenze di rumore

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k=[0:N-1]';                               % Vettore del tempo discreto
s=A*sqrt(1/(2*B))*cos(2*pi*F0*k+Theta);   % generazione del vettore segnale deterministico

h=waitbar(0,'Please wait...');
for kk=1:length(SNR)
    waitbar(kk/length(SNR),h); 
    
    for i=1:M
        Noise=sqrt(varW(kk))*randn(N,1);    % generazione del vettore Nx1 di rumore
        Y=s+Noise;                          % generazione del vettore Nx1 dei dati osservati 
        Yf=fft(Y,Nzp);
        Yf=Yf(1:Nzp/2+1);
        Periodogram=abs(Yf).^2./N;
        [maxp,indML]=max(Periodogram);
        fML(i)=2*B*(indML-1)*(1/Nzp);
        AML(i)=sqrt(2*B)*2.*abs(Yf(indML))./N;
        ThetaML(i)=angle(Yf(indML));
        RY0=mean(Y.^2);
        RY1=mean(Y(2:N).*Y(1:N-1));
        fMM(i)=(B/pi)*acos(RY1./RY0);
    end
    
    ErrA=AML-A;                          % Errore nella stima ML di A
    ErrTheta=ThetaML-Theta;              % Errore nella stima ML di Theta
    Errf0=fML-f0;                        % Errore nella stima ML di f0
    Errf0MM=fMM-f0;                      % Errore nella stima MM di f0
    MSE_A(kk)=mean(ErrA.^2);             % MSE nella stima ML di A
    MSE_Theta(kk)=mean(ErrTheta.^2);     % MSE nella stima ML di Theta 
    MSE_f0(kk)=mean(Errf0.^2);           % MSE nella stima ML di f0 
    MSE_f0MM(kk)=mean(Errf0MM.^2);       % MSE nella stima MM di f0 
end
close(h)

CRB_A=(A^2)./(N*SNR);
CRB_Theta=2*(2*N-1)./(SNR*N*(N+1));
CRB_f0=12*B^2./(pi^2*SNR*N*(N^2-1));

figure (1);
plot(SNRdB,10*log10(MSE_A),'LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(CRB_A),'r','LineWidth',2);
legend('MSE ML estimator of A ','CRB(A)');
title('MSE vs SNR_{in}');xlabel('SNR_{in} [dB]');ylabel('MSE [dB]')

figure (2);
plot(SNRdB,10*log10(MSE_Theta),'LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(CRB_Theta),'r','LineWidth',2);
legend('MSE ML estimator of  \Theta ','CRB(\Theta)');
title('MSE vs SNR_{in}');xlabel('SNR_{in} [dB]');ylabel('MSE [dB]')

figure (3);
plot(SNRdB,10*log10(MSE_f0),'LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(MSE_f0MM),'g','LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(CRB_f0),'r','LineWidth',2);
legend('MSE ML estimator of f_{0}','MSE MM estimator of f_{0}','CRB(f_{0})');
title('MSE vs SNR_{in}');xlabel('SNR_{in} [dB]');ylabel('MSE [dB]')


% Grafico di una realizzazione del segnale osservato

SNRdB=0;                    % Rapporto segnale-rumore in decibel [dB]
SNR=10^(SNRdB/10);          % SNR in scala lineare
N0=(A^2)/(2*B*SNR);         % Densità spettrale di Potenza monolatera del processo di rumore bianco  
varW=N0/2;                  % Vettore delle potenze di rumore
Noise=sqrt(varW)*randn(N,1);    
Y=s+Noise;
Nc=100;
Nc=min(Nc,N);
Yfig=Y(1:Nc);               % Grafico dei primi Nc campioni
tempo=k(1:Nc)*Tc;
figure(4)
stem(tempo,Yfig);
hold on
plot(tempo,Yfig,'r-','LineWidth',1);
legend('Y(t) - Segnale osservato in [0,T] - SNR=0 dB');
title('Segnale osservato in [0,T]');xlabel('Time [sec]');ylabel('Y(t)')

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
 