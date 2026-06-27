import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Patient, ProgressionPrediction } from '../models/patient.model';

@Injectable({
  providedIn: 'root',
})
export class PatientService {
  private baseUrl = 'http://127.0.0.1:5000/api/patients';

  constructor(private http: HttpClient) {}

  // GET: /api/patients?search=Petar&page=1
  getAllPatients(search?: string, page?: number): Observable<any> {
    let params = new HttpParams();
    if (search) params = params.set('search', search);
    if (page) params = params.set('page', page.toString());

    return this.http.get<any>(this.baseUrl, { params });
  }

  // GET: /api/patients/:id
  getPatientById(id: number): Observable<Patient> {
    return this.http.get<Patient>(`${this.baseUrl}/${id}`);
  }

  // POST: /api/patients
  createPatient(patient: Patient): Observable<Patient> {
    return this.http.post<Patient>(this.baseUrl, patient);
  }

  // PUT: /api/patients/:id
  updatePatient(id: number, data: Partial<Patient>): Observable<Patient> {
    return this.http.put<Patient>(`${this.baseUrl}/${id}`, data);
  }

  // DELETE: /api/patients/:id
  deletePatient(id: number): Observable<any> {
    return this.http.delete<any>(`${this.baseUrl}/${id}`);
  }

  // GET: /api/patients/:id/predict-progression (Veza sa GRU/XGBoost ansamblom)
  predictProgression(id: number): Observable<ProgressionPrediction> {
    return this.http.get<ProgressionPrediction>(`${`${this.baseUrl}/${id}/predict-progression`}`);
  }
}
