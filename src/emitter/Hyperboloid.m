classdef Hyperboloid < EmitterSim
    properties
        v_edges;
        z_edges;
        e_edges;
        right_sf; % scale factor for right BC location
    end

    methods
        function self = Hyperboloid(d, rc, alpha, h, ra, te, V0, ms, right_sf)
            self = self@EmitterSim(d, rc, alpha, h, ra, te, V0, ms);
            self.right_sf = right_sf;
            geo = self.constructGeometry(); % needs to set v,z,e edges
            self.solve(geo, self.v_edges, self.z_edges);
        end

        function geo = constructGeometry(self)
            % Calculated values
            curve_y_delta = self.rc - self.rc*sin(self.alpha);  % curved edge y-delta [m]
            curve_x_delta = self.rc*cos(self.alpha);            % curved edge x-delta [m]
            h_cone = self.h - curve_y_delta;                    % height of emitter cone [m]
            rb = curve_x_delta + h_cone*tan(self.alpha);        % Emitter base radius [m]

            % Get total arc lengths
            E1 = [0 self.d];
            E3 = [E1(1)+rb, 0];
            x_coord = linspace(E1(1),E3(1), 1000);
            H0 = 1/self.rc;
            eta0 = sqrt( (H0*self.d)/(H0*self.d + 1) );
            a = self.d * sqrt( (H0*self.d + 1)/(H0*self.d) );
            xi_coord = sqrt(1 + (x_coord/(a*sqrt(1-eta0^2))).^2);
            y_coord = a*eta0*xi_coord;
            ds = [0, sqrt(diff(x_coord).^2 + diff(y_coord).^2)];
            s = cumsum(ds);
            total_arc_length = s(end);

            % estimate curve discretization from mesh size
            N = round(total_arc_length/self.ms);      

            % Save BC (v and z) and emitter (e) edge numbers
            self.v_edges = 1:N+1;
            self.z_edges = N+3;
            self.e_edges = 1:N;

            % Emitter points [X,Y] : (0,0) is below emitter tip
            E1 = [0 self.d];
            E3 = [E1(1)+rb, 0];
            x_coord = linspace(E1(1),E3(1), N+1);
            H0 = 1/self.rc;
            eta0 = sqrt( (H0*self.d)/(H0*self.d + 1) );
            a = self.d * sqrt( (H0*self.d + 1)/(H0*self.d) );
            xi_coord = sqrt(1 + (x_coord/(a*sqrt(1-eta0^2))).^2);
            y_coord = a*eta0*xi_coord;

            % Construct discretized hyperboloid curve
            E = zeros(N,10);
            E(:,1) = 2;
            E(:,2) = x_coord(1:N);
            E(:,3) = x_coord(2:N+1);
            E(:,4) = y_coord(1:N);
            E(:,5) = y_coord(2:N+1);
            E(:,6) = 0;
            E(:,7) = 1;
            x_max = E3(1) * self.right_sf;   % Scene x boundary
            E3(2) = max(y_coord);
           
            % Bounding box (scene) points
            S1 = [0 0];
            S2 = [x_max S1(2)];
            S3 = [S2(1) E3(2)];

            % Construct decomposed geometry matrix manually (descg as an alternative)
            % Format: [segment_type, start_x, end_x, start_y, end_y, left_face_id, right_face_id, center_x, center_y, radius]'
            % segment_type: 1=round edge, 2=straight line

            % Traverse CW, so left=external (0), right=internal (1)
            air_top = [2, E3(1), S3(1), E3(2), S3(2), 0, 1, zeros(1,3)]';
            air_right = [2, S3(1), S2(1), S3(2), S2(2), 0, 1, zeros(1,3)]';
            bottom = [2, S2(1), S1(1), S2(2), S1(2), 0, 1, zeros(1,3)]';
            air_left = [2, S1(1), E1(1), S1(2), E1(2), 0, 1, zeros(1,3)]';
        
            geo = [E', air_top, air_right, bottom, air_left]; 
        end
    end
end