%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola il Mean Square Error (MSE) degli stimatori
% ML e MAP del segnale costante S aleatorio, distribuito come una v.a di
% Laplace, in funzione di SNRdB
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M=10^4;                      % Numero di prove Monte Carlo per ogni SNR
N=50;                        % Numero di campioni osservati per ogni stima
Lambda=1;                    % Parametro della Laplace
SNRdB=[-20:1:20];            % Vettore dei rapporti segnale-rumore in decibel [dB]
SNR=10.^(SNRdB./10);         % Vettore di SNR in scala lineare
varW=2*Lambda^2./SNR;        % Vettore delle potenze del rumore
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S=(exprnd(Lambda,M,1)).*(sign(rand(M,1)-0.5));    % Vettore delle M realizzazioni del segnale S

h=waitbar(0,'Please wait...');
for kk=1:length(SNR)
    waitbar(kk/length(SNR),h); 
    
    for i=1:M
        W=sqrt(varW(kk))*randn(N,1);    % generazione del vettore Nx1 di rumore
        Y=S(i)+W;                       % generazione del vettore Nx1 dei dati osservati 
        SML(i)=mean(Y);
        SMAP(i)=(SML(i)-2*Lambda/(N*SNR(kk)))*stepfun(abs(SML(i)),2*Lambda/(N*SNR(kk)));
    end
    
    ErrML=S-SML';                          % Errore nella stima ML di S
    ErrMAP=S-SMAP';                        % Errore nella stima MAP di S
    MSE_ML(kk)=mean(ErrML.^2);            % MSE nella stima ML di S
    MSE_MAP(kk)=mean(ErrMAP.^2);          % MSE nella stima MAP di S
end
close(h)

BCRB=Lambda^2./(1+N*SNR/2);

figure (1);
plot(SNRdB,10*log10(MSE_ML),'LineWidth',2);grid on;
hold on
plot(SNRdB,10*log10(MSE_MAP),'g','LineWidth',2);
plot(SNRdB,10*log10(BCRB),'r','LineWidth',2);

legend('ML estimator','MAP estimator', 'BCRB(s)');
title('MSE of ML and MAP estimates vs the BCRB');xlabel('SNR [dB]');ylabel('MSE [dB]')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 