function plotDecisionBoundary(theta, D, T)



figure;
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(D(:,1))-2,  max(D(:,1))+2];
for i= 1:size(theta,1)
    % Calculate the decision boundary line
    plot_y = (-1./theta(i,2)).*(theta(i,1).*plot_x + theta(i,3));
    
    
    
plot2dimdata(D,T, 'r*', 'k+',i);
hold on;
    % Plot, and adjust axes for better viewing
    
    plot(plot_x, plot_y);
    hold on;
%     %Legend, specific for the exercise
      legend('positive class', 'Negative class', 'Decision Boundary');
     % title('')
%     axis([30, 100, 30, 100])
    hold on;
end

end
