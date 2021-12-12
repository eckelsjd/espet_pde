classdef EmitterMS < EmitterSim

    properties
        e_edges;
    end

    methods
        function self = EmitterMS(d, rc, alpha, h, ra, te, ms)
            V0 = 1000; % [V]
            self = self@EmitterSim(d, rc, alpha, h, ra, te, V0, ms);
            v_edges = [1,2,3];    % Edge numbers with applied bias voltage
            z_edges = [5];  % Edge numbers with 0V bias (extractor)
            self.e_edges = [1,2];
            geo = self.constructGeometry();
            self.solve(geo, v_edges, z_edges);
        end

        function geo = constructGeometry(self)
            % Calculated values
            curve_y_delta = self.rc - self.rc*sin(self.alpha);  % curved edge y-delta [m]
            curve_x_delta = self.rc*cos(self.alpha);            % curved edge x-delta [m]
            h_cone = self.h - curve_y_delta;                    % height of emitter cone [m]
            rb = curve_x_delta + h_cone*tan(self.alpha);        % Emitter base radius [m]
            x_max = rb * 1.5;                                     % Scene x boundary
        
            % Emitter points [X,Y] : (0,0) is below emitter tip
            E1 = [0 self.d];
            Ecenter = [E1(1), E1(2)+self.rc];
            E2 = [E1(1)+curve_x_delta, E1(2)+curve_y_delta];
            E3 = [E1(1)+rb, E1(2)+self.h];
        
            % Bounding box (scene) points
            S1 = [0 0];
            S2 = [x_max S1(2)];
            S3 = [S2(1) E3(2)];

            % Construct decomposed geometry matrix manually (descg as an alternative)
            % Format: [segment_type, start_x, end_x, start_y, end_y, left_face_id, right_face_id, center_x, center_y, radius]'
            % segment_type: 1=round edge, 2=straight line

            % Traverse CW, so left=external (0), right=internal (1)
            round_edge = [1, E1(1), E2(1), E1(2), E2(2), 0, 1, Ecenter, self.rc]';
            cone_slant = [2, E2(1), E3(1), E2(2), E3(2), 0, 1, zeros(1,3)]';

            air_top = [2, E3(1), S3(1), E3(2), S3(2), 0, 1, zeros(1,3)]';
            air_right = [2, S3(1), S2(1), S3(2), S2(2), 0, 1, zeros(1,3)]';
            air_bottom = [2, S2(1), S1(1), S2(2), S1(2), 0, 1, zeros(1,3)]';
            air_left = [2, S1(1), E1(1), S1(2), E1(2), 0, 1, zeros(1,3)]';
        
            geo = [round_edge, cone_slant, air_top, air_right,...
                air_bottom, air_left]; 
        end
    end
end