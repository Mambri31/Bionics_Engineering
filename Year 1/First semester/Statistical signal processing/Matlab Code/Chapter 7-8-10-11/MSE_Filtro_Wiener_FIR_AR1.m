%
% Questo programma calcola e disegna la funzione di autocorrelazione di un processo AR(1)
% e l'MSE del filtro di Wiener FIR(P) per processi AR(1) in AWGN
%
clear all;
close all;


Pvect=[0:20]';           % ordine del filtro FIR di Wiener
varS=2;                  % Potenza di S[n]
gammadb=10;               % SNR in dB
gamma=10^(gammadb/10);   % SNR in scala lineare
varW=varS/gamma;         % Potenza di W[n]
varX=varS+varW;          % Potenza di X[n]

rhovect=[0.1 0.5 0.8 0.99]';      % One-lag correlation coefficient di S[n] processo AR(1)

for i=1:length(rhovect)
    
    rho=rhovect(i);
    
    for k=1:length(Pvect)
        P=Pvect(k);
        m=[0:P]';
        cs=varS*rho.^m;               % vettore della ACF di S[n]
        CW=varW*eye(P+1);             % matrice di covarianza di W[n]
        CS=varS*toeplitz((rho.^m)');  % matrice di covarianza di S[n]
        CX=CS+CW;                     % matrice di covarianza di X[n]
        CXinv=inv(CX);                % inversa della matrice di covarianza di X[n]
        h=CXinv*cs;                   % Risposta impulsiva del filtro di Wiener FIR(P)
        varEps(k,i)=10*log10(varS-h'*cs);
    end
end
%
figure(1)

plot(Pvect,varEps(:,1), 'b','Linewidth', 2);grid on;
hold on
plot(Pvect,varEps(:,2), 'g','Linewidth', 2);grid on;
plot(Pvect,varEps(:,3), 'r','Linewidth', 2);grid on;
plot(Pvect,varEps(:,4), 'magenta','Linewidth', 2);grid on;
legend('\rho=0.1', '\rho=0.5','\rho=0.8','\rho=0.99');
axis([0 P -12 -6]);
xlabel('Ordine del Filtro FIR (P)')
ylabel('MSE [dB]')
title('MSE del filtro di Wiener FIR(P) per processi S[n] di tipo AR(1)')
grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%