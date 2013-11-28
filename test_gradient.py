#Test Gradient
def test_gradient(GRAD, W,d,h1) : 
	W1_odd = W[0]
	W1_even = W[1]
	B1_odd = W[2]
	B1_even	= W[3]
	W2	= W[4]
	B2	= W[5]
	dwo1_Ei = GRAD[0]
 	dwe1_Ei = GRAD[1]
 	R1_odd = GRAD[2]
 	R1_even = GRAD[3]
	dw2_Ei = GRAD[4]
	R2 = GRAD[5]
	delta = 1e-8
	max_error = 0
	#Test wo1
	for j in range(0,h1) :
		for k in range(0,d) :
			derivative = (W1_odd[j][k]+delta - W1_odd[j][k])/delta
			error = dwo1_Ei[j][k] - derivative
			if error > max_error :
				max_error = error 
	#Test we1
	for j in range(0,h1) :
		for k in range(0,d) :
			derivative = (W1_even[j][k]+delta - W1_even[j][k])/delta
			error = dwe1_Ei[j][k] - derivative
			if error > max_error :
				max_error = error 
	#Test w2
	for j in range(0,h1) :
		derivative = (W2[j][0]+delta - W2[j][0])/delta
		error = dw2_Ei[j][0] - derivative
		if error > max_error :
			max_error = error
	#Test B1_odd
	for j in range(0,h1) :
		derivative = (B1_odd[j][0]+delta - B1_odd[j][0])/delta
		error = R1_odd[j][0] - derivative
		if error > max_error :
			max_error = error
	#Test B1_even
	for j in range(0,h1) :
		derivative = (B1_even[j][0]+delta - B1_even[j][0])/delta
		error = R1_even[j][0] - derivative
		if error > max_error :
			max_error = error
	#Test B1_even
	for j in range(0,h1) :
		derivative = (B2+delta - B2)/delta
		error = R2 - derivative
		if error > max_error :
			max_error = error
	return max_error
