%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola l'istogramma degli stimatori ML di ampiezza e fase 
% di un oscillazione cosinusoidale immersa in rumore AWGN 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nbin=100;
M=10^6;                      % Numero di prove Monte Carlo per ogni SNR
A=3;                         % Ampiezza del segnale
Theta=2*pi/3;                % Fase iniziale in radianti
T=10;                        % Intervallo di osservazione
SNRdB=0;                    % Vettore dei rapporti segnale-rumore in decibel [dB]
SNR=10.^(SNRdB/10);          % Vettore di SNR in scala lineare
N0=((A^2)*T)/(2*SNR);        % Densità Spettrale di Potenza monolatera del rumore
varW=N0/2;                   % Vettore delle potenze di rumore
S1=A*sqrt(T/2)*cos(Theta);   % generazione della componente di segnale deterministico - componente 1
S2=-A*sqrt(T/2)*sin(Theta);  % generazione della componente di segnale deterministico - componente 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    W1=sqrt(varW)*randn(M,1);           % generazione del vettore di rumore - componente 1
    W2=sqrt(varW)*randn(M,1);           % generazione del vettore di rumore - componente 2
    X1=S1+W1;                           % generazione del vettore dati osservati - componente 1
    X2=S2+W2;                           % generazione del vettore dati osservati - componente 2
    Ahat=sqrt(2/T)*sqrt(X1.^2+X2.^2);   % Stima ML di A
    Thetahat=-atan2(X2,X1);             % Stima ML di Theta
    CRB_A=(A^2)./(2*SNR);
    CRB_Theta=1./(2*SNR);

    [countA,a0]=hist(Ahat,Nbin);       % esegue i conteggi
    DeltaA=a0(2)-a0(1);                % calcola larghezza bin
    ddpAhat=countA/DeltaA/M;           % istogramma normalizzato

    [countT,t0]=hist(Thetahat,Nbin);   % esegue i conteggi
    DeltaT=t0(2)-t0(1);                % calcola larghezza bin
    ddpThat=countT/DeltaT/M;           % istogramma normalizzato
    

    figure (1);
    bar(a0,ddpAhat,'b');                       % grafico istogramma normalizzato
    hold on
    grid on
    plot(a0,normpdf(a0,A,sqrt(CRB_A)),'r-','LineWidth',2)% grafico ddp Gaussiana
    legend('ML Estimate of A - Histogram','Asymptotic pdf');
    title('ML Estimate of A - Histogram');xlabel('a');ylabel('Histogram and asymptotic pdf ')
    axis([0 14 0 0.25]);
    
    figure (2);
    bar(t0,ddpThat,'b');                       % grafico istogramma normalizzato
    hold on
    grid on
    plot(t0,normpdf(t0,Theta,sqrt(CRB_Theta)),'r-','LineWidth',2)% grafico ddp Gaussiana
    legend('ML Estimate of \Theta - Histogram','Asymptotic pdf');
    title('ML Estimate of \Theta - Histogram');xlabel('\Theta');ylabel('Histogram and asymptotic pdf ')
    axis([-pi pi 0 0.6]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 