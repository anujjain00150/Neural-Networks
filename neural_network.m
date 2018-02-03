%%====================== MACHINE LEARNING Assignment 6 =========================
%
%   Team Members
%   Pavan Siva Kumar Amarapalli
%   Venkata Praneeth Bavirisetty
%   Shiva Vamsi Gudivada
%   Anuj Jain
%  
%
%%=========================== Initialization ==============================

clear all;
close all;
clc;
%...............................Part I.................

fprintf('\n	Ploting error as a function of iterations.\n');
data =-2 + 6*rand(100,2);
%syms x1 x2;
[r, c] = size(data);
%y = x1 + 2*x2 -2;

for i = 1:r
if  (data(i,1) + 2*data(i,2) -2)> 0, target(i,1) =1; else target(i,1)=0; end
end

k =1;  % Rate of change of eta
eta = 0.03;
error = 0.004;
epochs =2000;


[w, iterations1, e , w_iter]=DeltaBatchRuleTraining(data, target, eta, error, epochs);
plot(1:(iterations1),e);
xlabel('iterations');
ylabel('error');
iterations1
answer = [data ones(100,1)]*w(:);

accuracy = mean(double(answer>0 == target>0))*100


fprintf('Press Enter for part 1(b).\n');
pause;

%.....................................Part II.......................
fprintf('\n Decision Boundary for 5, 10, 50 and 100 iterations.\n');
plotDecisionBoundary(w_iter, data, target);
title('Decision Boundary for 5, 10, 50 and 100 iterations');
hold off;

fprintf('Press Enter for part 1(c).\n');
pause;

%.....................................Part III.....................

fprintf('\n Analyze differnt learning rates.\n');
i = 0;
figure;
for eta = [0.0003 0.003 0.03 0.1]
    i =i+1;
    rate(i) = eta;
[w, iterations, e , w_iter]=DeltaBatchRuleTraining(data(1:80,:), target(1:80,1), eta, error, epochs);
subplot(2,2,i);
plot(1:(iterations),e);
xlabel('iterations');
ylabel('error(with different learning Rates)');
title(rate(i) );
answer = [data ones(100,1)]*w(:);
accuracy = mean(double(answer>0 == target>0))*100
end

%suptitle('Learning with different rates' );
hold off;
fprintf('For better learning rates we get high accuracy.\n');

fprintf('Press Enter for part 1(d).\n');
pause;

%....................................Part IV........................

fprintf('\n Implementing Delta rule in incremental fashion.\n');
[w, iterations2, e ]=DeltaRuleTraining(data, target, eta, error, epochs,k);
answer = [data ones(100,1)]*w(:);
%plot(1:(iterations),e);
%title('Error');
accuracy = mean(double(answer>0 == target>0))*100
fprintf('Iterations in Batch mode is %f. \n',iterations1);
fprintf('Iterations in incremental Delta Rule is %f . \n', iterations2);
fprintf('\n It is clearly seen by this incremental \n delta rule works fast.\n')
fprintf('Press Enter for part 2(a).\n');
pause;

%....................................Part V........................
fprintf('\n Implementing Decaying learning rates.\n');
error = 0.003;
k=0.9;  % Decaying Rate
eta = 1;
[w, iterations3, e ]=DeltaRuleTraining(data, target, eta, error, epochs,k);
answer = [data ones(100,1)]*w(:);
accuracy = mean(double(answer>0 == target>0))*100
fprintf('Iterations in Batch mode is %f. \n',iterations3);
fprintf('It can be clearly seen decaying learning \n rates slows the algorithm.\n')
fprintf('Press Enter for part 2(b).\n');
pause;

%....................................Part VI........................
fprintf('\n Implementing Adaptive learning rates.\n');
error = 0.5;
D = 1.05;
d=0.94;  % Decaying Rate
eta = 0.08;
t = 0.02;
figure;
[w, iterations3, e ]=AdaptiveDeltaRuleTraining(data, target, eta, error, epochs, d, D, t);
iterations3
xlabel('iterations');
ylabel('Adaptive learning rate');

answer = [data ones(100,1)]*w(:);
accuracy = mean(double(answer>0 == target>0))*100

fprintf('Press Enter for part 3.\n');
pause;
%....................................Part VII........................
fprintf('\n Implementing Gradient Descent on Sigmoid data.\n');
k =1;  % Rate of change of eta
eta = 0.03;
error = 0.004;
%load('toydatax.mat');
% data2(:,1) = toydatax(:,1).^2 + toydatax(:,1);
% data2(:,2) = toydatax(:,2).^2 + toydatax(:,2);
% target2(:,1) = toydatay(:);
% target2 = target2>0;
data3 = gen_sigmoid_classes(100);
data2(:,1) = data3(:,1).^2 + data3(:,1);
data2(:,2) = data3(:,2).^2 + data3(:,2);
target2(:,1) = data3(:,3);
target2 = target2>0;

[w, iterations3, e ]=DeltaRuleTraining(data2, target2, eta, error, epochs,k);
iterations3
 answer = [data2 ones(100,1)]*w(:);
 accuracy = mean(double(answer>0 == target2>0))*100
plotDecisionBoundary(w, data2, target2);
