%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo programma grafica la ddp del massimo di N v.a. indipendenti ed uniformemente
% distribuite tra 0 e Theta e poi grafica gli MSE dello stimatore ML e dello
% stimatore MM di Theta
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

Theta=1;
Nvect=[1 5 10 20 100]';
x=[0:0.01:Theta]';

for i=1:length(Nvect)
    N=Nvect(i);
    ddpmat(:,i)=N/Theta*(x./Theta).^(N-1);

end
figure (1);
plot(x,ddpmat,'LineWidth',2);grid on;
hold on
legend('N=1','N=5','N=10','N=20','N=100');
title('Probability Density Function of the ML Estimate of  \Theta');xlabel('x_M');ylabel('Pdf ML Estimator')
axis([0 Theta 0 70])

Nvect=2.^[0:10]';
Theta=1;
MSEML=2*Theta^2./((Nvect+1).*(Nvect+2));
MSEMM=Theta^2./(3*Nvect);
figure (2);
loglog(Nvect,MSEML,'b','LineWidth',2);grid on;
hold on
loglog(Nvect,MSEMM,'r','LineWidth',2);grid on;
legend('MSE ML Estimator','MSE MM Estimator');
title('MSE of the ML and MM Estimators vs. Data Size N');xlabel('Data size (N)');ylabel('MSE')
axis([1 1000 10^-5 1])
