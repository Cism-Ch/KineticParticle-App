import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { ParticleConfig, ParticleMode } from '../types';

interface ParticleSceneProps {
  config: ParticleConfig;
}

const dummy = new THREE.Object3D();
const tempVec = new THREE.Vector3();

export const ParticleScene: React.FC<ParticleSceneProps> = ({ config }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  
  // Initialize particle data
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < config.count; i++) {
      const t = Math.random() * 100;
      const factor = 20 + Math.random() * 100;
      const speed = 0.01 + Math.random() / 200;
      const x = Math.random() * 100 - 50;
      const y = Math.random() * 100 - 50;
      const z = Math.random() * 100 - 50;
      
      temp.push({ t, factor, speed, x, y, z, mx: 0, my: 0, mz: 0, originalColor: new THREE.Color() });
    }
    return temp;
  }, [config.count]);

  // Handle color updates smoothly
  useEffect(() => {
    if (!meshRef.current) return;
    const color = new THREE.Color(config.color);
    for (let i = 0; i < config.count; i++) {
      meshRef.current.setColorAt(i, color);
    }
    meshRef.current.instanceColor!.needsUpdate = true;
  }, [config.color, config.count]);

  useFrame((state) => {
    if (!meshRef.current) return;

    const time = state.clock.getElapsedTime();
    const { mode, speed } = config;

    particles.forEach((particle, i) => {
      let { x, y, z, t, factor, speed: pSpeed } = particle;

      // Logic based on modes
      switch (mode) {
        case ParticleMode.FLOAT:
          // Perlin-ish noise movement
          t = particle.t += speed * 0.5;
          const a = Math.cos(t) + Math.sin(t * 1) / 10;
          const b = Math.sin(t) + Math.cos(t * 2) / 10;
          const s = Math.cos(t);
          particle.mx += (x * a - particle.mx) * 0.1;
          particle.my += (y * b - particle.my) * 0.1;
          particle.mz += (z * s - particle.mz) * 0.1;
          
          dummy.position.set(
            x + Math.sin(time + x) * 2,
            y + Math.cos(time + y) * 2,
            z + Math.sin(time + z) * 2
          );
          break;

        case ParticleMode.GATHER:
          // Attract to center
          particle.mx += (0 - particle.mx) * speed * 0.1;
          particle.my += (0 - particle.my) * speed * 0.1;
          particle.mz += (0 - particle.mz) * speed * 0.1;
          
          // Lerp current position towards zero
          dummy.position.set(
            x * 0.1 + Math.sin(time * 10 + i) * 2,
            y * 0.1 + Math.cos(time * 10 + i) * 2,
            z * 0.1
          );
          break;

        case ParticleMode.EXPLODE:
          // Push away from center
          dummy.position.set(
            x * (1 + Math.sin(time) * 0.5 + 1.5),
            y * (1 + Math.cos(time) * 0.5 + 1.5),
            z * (1 + Math.sin(time * 0.5) * 0.5 + 1.5)
          );
          break;

        case ParticleMode.SWIRL:
           // Vortex
           const angle = time * speed * 2 + (i * 0.01);
           const radius = 15 + Math.sin(time + i) * 5;
           dummy.position.set(
             Math.cos(angle) * radius,
             (y + Math.sin(time) * 10) * 0.5,
             Math.sin(angle) * radius
           );
           dummy.rotation.y = angle;
           break;
           
        case ParticleMode.RAIN:
           // Falling down
           const fallSpeed = 50 * speed;
           let newY = y - (time * fallSpeed) % 100;
           if (newY < -50) newY += 100;
           dummy.position.set(x, newY, z);
           dummy.scale.set(1, 5, 1);
           break;
      }

      if (mode !== ParticleMode.SWIRL && mode !== ParticleMode.RAIN) {
          dummy.rotation.set(Math.sin(time), Math.cos(time), 0);
          dummy.scale.setScalar(1);
      }
      
      if (mode === ParticleMode.RAIN) {
         // Reset scale for other modes in next frame if needed, but handled by switch
      } else {
         dummy.scale.setScalar(config.size);
      }

      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, config.count]}>
      <dodecahedronGeometry args={[0.2, 0]} />
      <meshPhongMaterial 
        color={config.color} 
        emissive={config.color} 
        emissiveIntensity={0.5}
        specular="#ffffff"
        shininess={100}
      />
    </instancedMesh>
  );
};
