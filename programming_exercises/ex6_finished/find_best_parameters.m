%% use the cross validation set Xval, yval to determine the best C and parameter to use

%After you have determined the best C and sigma parameters to use, you
%should modify the code in dataset3Params.m,filling in the best parameters
% 

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

min_error = 1;
C = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma = [0.01,0.03,0.1,0.3,1,3,10,30];

for i = 1:8
    for j = 1:8
        
        % Train the SVM
        model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
        
        %predict
        predy = svmPredict(model,Xval);
        
        %error rate
        error = mean(double(predy~= yval));
        
        if error < min_error
            best_C = C(i);
            best_sigma = sigma(j);
            min_error = error;
        end
    end
end

fprintf('best_C is %f,best_sigma is %f.\n',best_C,best_sigma); 


