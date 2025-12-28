<img width="1533" height="849" alt="image" src="https://github.com/user-attachments/assets/d9d06825-4d4b-4fe8-802d-881ee3d53f10" />


Program that generates .dat files for Grid Sag zemax surface type. It can generate medium frequency errors (MSFE). Useful for simulating MSFE error on aspheric mirrors.

If your surface uses only conic, you can just select surface type grid sag, which supports it. If your surface type is something like even asphere or else, to add this error to the surface you can do this: 
1) Add grid sag surface before your surface
2) Load .dat file
3) Make this surface composite surface. That way sag of this grid surface will be added to the next surface.
