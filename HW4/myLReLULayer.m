classdef myLReLULayer < nnet.layer.Layer

    properties (Learnable)
        % Layer learnable parameters
        
        % Scaling coefficient
        Alpha
    end
    
    methods
        function layer = myLReLULayer(numChannels, name) 
            % Create my layer with numChannels channels

            % Set layer name
            if nargin == 2
                layer.Name = name;
            end

            % Set layer description
            layer.Description = ...
                ['myLReLULayer with ', num2str(numChannels), ' channels'];
            
            % Initialize parameter
            layer.Alpha = ones([1 1 numChannels]) .* (0.01);
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer and output the result
            
            Z = max(0, X) + layer.Alpha .* min(0, X);
        end
        
        function [dLdX, dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            
            dLdX = layer.Alpha .* dLdZ;
            dLdX(X>0) = dLdZ(X>0);
            
            dLdAlpha = single(zeros(size(layer.Alpha)));
        end
    end
end