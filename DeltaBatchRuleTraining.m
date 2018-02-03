function [w, iterations, e, w_iter]=DeltaBatchRuleTraining(Data, Target, eta, error, epochs)
%% Invoke as: [w, iterations, e] = DeltaRuleTraining(Data, Target, eta, error, epochs)
%% implements the delta  rule;
%% Input:
%%  Data is a matrix N x P data points/vectors
%%  Target is vector N x 1 of target values (true output) corresponding to the data points
%%  eta: learning rate; 
%%  error : desired approximation error;
%%  epochs: threshold on the number of epochs (iterations through the whole
%% data set)
%% Output:
%%  w is a vector of dimension P+1 x 1, where w_i is the weight for dimension i of a data point,
%%     for i=1:P, extended with weight w0 for the bias (input = 1)
%%  iterations = MIN{is the number of iterations taken to reach error threshold e, epochs}
%%  e: error threshold

[rd, cd]=size(Data);
[rt, ct]=size(Target);
if rt ~= rd
    error('num data points not equal to num target');
else
 w=rand(1,cd+1);
 %w=zeros(1,cd+1);
 iterations=0;
err=error;
i = 0;
Data = [Data  ones(rd,1)];
while err >= error &&  iterations <= epochs
 iterations=iterations+1;
 wrong=0;
 
 out = Data*w';
 out = sigmoid (out);

     err=sum((Target- out).^2)/(2*rd);

     deltaw=eta*(Target-out)'*Data;
     w=w+deltaw;
     
     if err>0
         wrong = wrong+1;
     end
     e(iterations)=err;
     %e = sum(err)/rd;
     
     if iterations == 5 || iterations == 10 ||iterations == 50 ||iterations == 100
         i= i+1;
         w_iter(i,:) = w;
    end
     
end
 end  % for i=1:rd 
% total error for perceptron
% e=wrong/rd;

% error for delta rule
end