import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatTabsModule } from '@angular/material/tabs';

export interface OftalmoloskiPregled {
  id: number;
  datum: string;
  posetaId: string; // Oznaka posete iz GRAPE baze (npr. 'M00', 'M12', 'M24')
  tip: 'Inicijalni' | 'Kontrola';

  // Klinički parametri (IOP)
  iopOD: number;
  iopOS: number;

  // Strukturni parametri (Fundus / Optički disk)
  cdrOD: number;
  cdrOS: number;
  slikaOD: string;
  slikaOS: string;
  maskaOD: string;
  maskaOS: string;

  // Funkcionalni parametri (Vidno polje - Perimetrija iz GRAPE dataseta)
  mdOD: number; // Mean Deviation desno (u dB)
  mdOS: number; // Mean Deviation levo (u dB)
  psdOD: number; // Pattern Standard Deviation desno
  psdOS: number; // Pattern Standard Deviation levo

  nalazLekara: string;
}

@Component({
  selector: 'app-patient-detail',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatListModule,
    MatTabsModule,
  ],
  templateUrl: './patient-detail.html',
  styleUrls: ['./patient-detail.scss'],
})
export class PatientDetail {
  // Podaci o pacijentu
  pacijent = {
    ime: 'Marko Nikolić',
    godiste: '1985',
    jmbg: '1204985710023',
    dijagnoza: 'Glaucoma chronicum simplex (H40.1)',
    napomena: 'Pacijent neredovan sa terapijom kapima. Pratiti CDR strogo.',
  };

  // Istorija pregleda ovog pacijenta
  pregledi: OftalmoloskiPregled[] = [
    {
      id: 3,
      datum: '31.05.2026.',
      posetaId: 'M24 (Nakon 2 godine)',
      tip: 'Kontrola',
      iopOD: 22,
      iopOS: 16,
      cdrOD: 0.65,
      cdrOS: 0.4, // Progresija ekskavacije na desnom oku
      mdOD: -6.42,
      mdOS: -1.2, // Vidno polje na desnom oku degradira (broj ide u veći minus)
      psdOD: 4.85,
      psdOS: 1.5,
      slikaOD: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400',
      slikaOS: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400',
      maskaOD: 'rgba(255, 0, 0, 0.4)', // Jača crvena maska jer je oštećenje veće
      maskaOS: 'rgba(0, 255, 0, 0.1)',
      nalazLekara:
        'Poseta M24. Strukturna progresija na OD jasna. CDR je porastao na 0.65. Funkcionalno, vidno polje beleži dodatni pad (MD -6.42 dB). Pojačati lokalnu terapiju analogom prostaglandina.',
    },
    {
      id: 2,
      datum: '05.05.2025.',
      posetaId: 'M12 (Nakon 1 godine)',
      tip: 'Kontrola',
      iopOD: 23,
      iopOS: 17,
      cdrOD: 0.61,
      cdrOS: 0.4,
      mdOD: -4.85,
      mdOS: -1.15,
      psdOD: 3.9,
      psdOS: 1.45,
      slikaOD: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400',
      slikaOS: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400',
      maskaOD: 'rgba(255, 0, 0, 0.25)',
      maskaOS: 'rgba(0, 255, 0, 0.1)',
      nalazLekara:
        'Poseta M12. Parametri umereno stabilni, ali uočava se rani trend produbljivanja skotoma u donjem nazalnom kvadrantu na desnom oku.',
    },
    {
      id: 1,
      datum: '15.04.2024.',
      posetaId: 'M00 (Bazni pregled)',
      tip: 'Inicijalni',
      iopOD: 25,
      iopOS: 19,
      cdrOD: 0.58,
      cdrOS: 0.38,
      mdOD: -3.1,
      mdOS: -1.05,
      psdOD: 2.8,
      psdOS: 1.3,
      slikaOD: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400',
      slikaOS: 'https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400',
      maskaOD: 'rgba(255, 165, 0, 0.3)', // Narandžasto upozorenje na početku
      maskaOS: 'rgba(0, 255, 0, 0.1)',
      nalazLekara:
        'Uključivanje u GRAPE praćenje. Asimetrija optičkih diskova. Desni disk sumnjiv sa temporalnim stanjivanjem neuroretinalnog prstena. Započeta terapija beta-blokatorima.',
    },
  ];

  // Selektovani pregled (podrazumevano najnoviji)
  izabraniPregled: OftalmoloskiPregled = this.pregledi[0];

  // Stanje prekidača za masku
  prikaziMasku: boolean = false;

  selektujPregled(pregled: OftalmoloskiPregled) {
    this.izabraniPregled = pregled;
  }

  toggleMaska() {
    this.prikaziMasku = !this.prikaziMasku;
  }
}
