
## ---- Unit conversion ----

def ft_to_m(ft_value):
    # Converts from ft to m 
    m_value = ft_value/3.2808
    return m_value

def m_to_ft(m_value):
    # Converts from ft to m 
    ft_value = m_value*3.2808
    return ft_value
        
def kt_to_ms(kt_value):
    # Converts from knots to m/s
    ms_value = kt_value*0.5144
    return ms_value

def rad_to_deg(rad_value):
    # Converts from radians to degrees
    deg_value = rad_value * 180/np.pi
    return deg_value

def deg_to_rad(deg_value):
    # Converts from radians to degrees
    rad_value = deg_value * np.pi/180
    return

def lb_to_kg(lb_value):
    # Converts from lbs to kgs
    kg_value = lb_value*0.453592
    return kg_value

def slgft2_tokgm2(slgft2_value):
    # Converts from slugs*ft^2 to kg*m^2
    kgm2_value = slgft2_value*1.35581795
    return kgm2_value

def lbft2_to_pa(lbft2_value):
    # Converts from lb/ft^2 to Pa
    pa_value = lbft2_value*47.880172
    return pa_value

def kmh_to_ms(kmh_value):
    # Converts from km/h to m/s
    ms_value = kmh_value/3.6
    return ms_value