local CrossProduct, parent
if not pcall(function()
            CrossProduct, parent = torch.class('nn.CrossProduct', 'nn.Module')
         end) then
    CrossProduct = nn.CrossProduct
    parent = nn.Module
end

function CrossProduct:__init()
    parent.__init(self)
    self.gradInput = {}
end

function CrossProduct:updateOutput(input)
	--[[expect last dimension has size 3 for both inputs]]
	local A, B = unpack(input)
	self.output = torch.cross(A, B, A:nDimension())
    return self.output
end

function CrossProduct:updateGradInput(input, gradOutput)
	--[[
	u x v =
	u2 v3 - u3 v2
	u3 v1 - u1 v3
	u1 v2 - u2 v1

	= Vu =
	0	v3		-v2	|	u1
	-v3	0		v1	|	u2
	v2	-v1		0	|	u3
	= Uv =
	0	-u3		u2	|	v1
	u3	0		-u1	|	v2
	-u2	u1		0	|	v3

	if G has row vectors gradients
	G d(u x v)/dv = GU = (U^T G^T)^T = - (u x G1 | u x G2 | ... | u x Gn)^T
									= (G1 x u | G2 x u | ... | Gn x u)^T
	where Gi are rows of G

	similarly
	G d(u x v)/du = GV = (V^T G^T)^T = - (G1 x v | G2 x v | ... | Gn x v)^T
									= (v x G1 | v x G2 | ... | v x Gn)^T
	]]
	local A, B = unpack(input)
	for i = 1, 2 do
		self.gradInput[i] = self.gradInput[i] or A.new()
		self.gradInput[i]:type(A:type())
	end
	-- d/du
	self.gradInput[1]:cross(B, gradOutput, B:nDimension())
	self.gradInput[2]:cross(gradOutput, A, A:nDimension())
    return self.gradInput
end

function CrossProduct:type(type, tensorCache)
    self.gradInput = {}
    return parent.type(self, type, tensorCache)
end